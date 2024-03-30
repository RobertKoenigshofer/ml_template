import torch


def collate_fn(batch):
    num_elements = len(batch[0])
    result = [ [] for _ in range(num_elements)]
    for sample in batch:
        for i, elem in enumerate(sample):
            result[i].append(elem)
    for i in range(len(result)):
        result[i] = torch.stack(result[i])
    return result


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers=1, collate_fn=collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        self._index = None
        self._indices = None
    
    def __iter__(self):
        self._index = 0
        if self.shuffle:
            self._indices = torch.split(torch.randperm(len(self.dataset)), self.batch_size)
        else:
            self._indices = torch.split(torch.arange(len(self.dataset)), self.batch_size)
        return self

    def __next__(self):
        if self._index < len(self._indices):
            batch = [self.dataset[idx] for idx in self._indices[self._index]]
            self._index += 1
            return self.collate_fn(batch)
        else:
            raise StopIteration