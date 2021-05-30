import os
import multiprocessing
from multiprocessing import Process
try:
    import nvidia_smi
except:
    pass
    
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
multiprocessing.set_start_method('fork')

def scan_run(self):

    '''The high-level management of the scan procedures
    onwards from preparation. Manages round_run()'''

    from tqdm import tqdm

    from .scan_prepare import scan_prepare
    self = scan_prepare(self)

    # initiate the progress bar
    self.pbar = tqdm(total=len(self.param_object.param_index),
                     disable=self.disable_progress_bar)

    # the main cycle of the experiment
    processes = []
    total = len(self.param_object.param_index)
    count = 0
    num_gpus = 0
    try:
        nvidia_smi.nvmlInit()
        num_gpus = nvidia_smi.nvmlDeviceGetCount()
        nvidia_smi.nvmlShutdown()
    except:
        pass
    gpu_tickets = [0] * num_gpus
    
    while True:

        if self.use_multiprocessing:
            while len(processes) >= self.max_processes:
                done = multiprocessing.connection.wait([x.sentinel for x in processes])
                for p in filter(lambda x: x.sentinel in done, processes):
                    gpu_id = -1
                    try:
                        gpu_id = getattr(p, 'gpu_id')
                    except:
                        pass
                    if gpu_id != -1:
                        gpu_tickets[gpu_id] -= 1
                    processes.remove(p)
                    count += 1
                    print(f"completed {count} of {total} runs ({(100.0 * count) / total : 0.2f}%)")
                    self.pbar.update(1)
                
        # get the parameters
        self.round_params = self.param_object.round_parameters()

        # break when there is no more permutations left
        if self.round_params is False:
            break
        
        # otherwise proceed with next permutation
        from .scan_round import scan_round
        if self.use_multiprocessing:
            # allocate a GPU
            gpu_id = 0
            try:
            	# reuse GPUs
                gpu_id = gpu_tickets.index(min(gpu_tickets))
                gpu_tickets[gpu_id] += 1
            except:
                # no GPU available
                gpu_id = -1
                pass
            p = Process(target=scan_round, args=(self,))
            self.round_params['GPU_ID'] = gpu_id
            setattr(p, 'gpu_id', gpu_id)
            processes.append(p)
            p.start()
        else:
            self = scan_round(self)
            self.pbar.update(1)

    if self.use_multiprocessing:
        while len(processes) > 0:
            done = multiprocessing.connection.wait([x.sentinel for x in processes])
            for p in filter(lambda x: x.sentinel in done, processes):
                gpu_id = -1
                try:
                    gpu_id = getattr(p, 'gpu_id')
                except:
                    pass
                processes.remove(p)
                if gpu_id != -1:
                    gpu_tickets[gpu_id] -= 1
                count += 1
                print(f"completed {count} of {total} runs ({(100.0 * count) / total : 0.2f}%)")
                self.pbar.update(1)

    # close progress bar before finishing
    self.pbar.close()

    # finish
    from ..logging.logging_finish import logging_finish
    self = logging_finish(self)

    from .scan_finish import scan_finish
    self = scan_finish(self)

    
    
    