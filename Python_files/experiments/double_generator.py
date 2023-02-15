# %%
import numpy as np

class BatchGenerator:

    def __init__(self, inp, ratio = 0.7):

        self.train_rows = int(np.ceil(len(inp) * ratio))
        self.test_rows = len(inp) - self.train_rows

        self.train_data = inp[:self.train_rows]
        self.test_data = inp[self.train_rows:]

        self.train_idx = 0
        self.test_idx = 0

    def has_train_data(self) -> bool:
        if self.train_idx >= self.train_rows:
            return False
        return True

    def has_test_data(self) -> bool:
        if self.test_idx >= self.test_rows:
            return False
        return True

    def train_batch(self) -> int:
        """Return a batch from the training set

        Returns:
            _type_: _description_
        """
        batch = self.train_data[self.train_idx]
        self.train_idx += 1

        return batch

    def test_batch(self) -> int:
        """Return a batch from the testing set

        Returns:
            _type_: _description_
        """
        batch = self.test_data[self.test_idx]
        self.test_idx += 1

        return batch

class TrainGenerator:

    def __init__(self, base_generator: BatchGenerator):
        self.base_generator = base_generator

    def __iter__(self):

        return self

    def __next__(self):
        if self.base_generator.has_train_data():
            return self.base_generator.train_batch()

        raise StopIteration

class TestGenerator:

    def __init__(self, base_generator: BatchGenerator):
        self.base_generator = base_generator

    def __iter__(self):

        return self

    def __next__(self):
        if self.base_generator.has_test_data():
            return self.base_generator.test_batch()

        raise StopIteration


def make_generators(inp: list[int]):

    batch_generator = BatchGenerator(inp)

    train_generator = TrainGenerator(batch_generator)
    test_generator = TestGenerator(batch_generator)

    return train_generator, test_generator


# %%

train_generator, test_generator = make_generators([1,2,3,4,5,6,7])

# %%

for b in train_generator:
    print(b)

# %%

for b in test_generator:
    print(b)

# %%
