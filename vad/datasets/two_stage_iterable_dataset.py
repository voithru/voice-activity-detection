import math
import random

from more_itertools import ichunked
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import get_worker_info
from torch.utils.data.dataset import IterableDataset

from vad.data_models.vad_data import VADDataList


class TwoStageIterableDataset(IterableDataset):
    variable_length_fields = {}

    def __init__(
        self,
        map_dataset,
        data_list: VADDataList,
        chunk_size=1,
        num_workers=0,
        **kwargs,
    ):
        self.map_dataset = map_dataset
        self.data_list = data_list
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            workload = self.data_list.pairs
        else:
            per_worker = int(math.ceil(len(self.data_list.pairs) / worker_info.num_workers))
            worker_id = worker_info.id
            workload_start = per_worker * worker_id
            workload_end = workload_start + per_worker
            workload = self.data_list.pairs[workload_start:workload_end]

        random.shuffle(workload)
        for workload_chunk in ichunked(workload, n=self.chunk_size):
            dataset = self.map_dataset(
                data_pairs=workload_chunk, num_workers=self.num_workers, **self.kwargs
            )
            sampler = RandomSampler(data_source=dataset)

            for sample in sampler:
                yield dataset[sample]
