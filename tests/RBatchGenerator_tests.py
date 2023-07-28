import unittest
import ROOT
import numpy as np

np.random.seed(1234)


def create_dataset(num_events, num_features, num_outputs, dtype=np.float32):
    x = np.random.normal(0.0, 1.0, (num_events, num_features)).astype(dtype=dtype)
    if num_outputs == 1:
        y = np.random.normal(0.0, 1.0, (num_events)).astype(dtype=dtype)
    else:
        y = np.random.choice(
            a=range(num_outputs),
            size=(num_events),
            p=[1.0 / float(num_outputs)] * num_outputs,
        ).astype(dtype=dtype)
    return x, y


def createRDataFrame(tree_name: str, file_name: str, num_events: int, num_columns: int):
    ROOT.gRandom.SetSeed(1)

    df = ROOT.RDataFrame(num_events)

    for i in range(num_columns):
        df = df.Define(f"r_{i}", "gRandom->Gaus(0,1)")

    df.Snapshot(tree_name, file_name)


def _test_RBatchGenerator_NumPy(tree_name: str, file_name: str, batch_size: int, chunk_size: int):
    print(f"Starting basic NumPy Test")

    train_generator, validation_generator = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        tree_name, file_name, batch_size, chunk_size
    )

    for b in train_generator:
        print(b.shape)
        print(b)

    for b in validation_generator:
        print(b.shape)


def _test_RBatchGenerator_NumPy_filtered(tree_name: str, file_name: str, batch_size: int, chunk_size: int):
    print(f"Starting filtered NumPy Test")
    filters = ["r_1 > 0.3", "r_2 < 0.8", "r_3 > 0.1"]

    train_generator, validation_generator = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        tree_name, file_name, batch_size, chunk_size, filters=filters
    )

    for b in train_generator:
        print(b.shape)
        print(b)

    for b in validation_generator:
        print(b.shape)


def _test_RBatchGenerator_PyTorch(tree_name: str, file_name: str, batch_size: int, chunk_size: int):
    print(f"Starting basic PyTorch Test")
    train_generator, validation_generator = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
        tree_name, file_name, batch_size, chunk_size
    )

    for b in train_generator:
        print(b.shape)

    for b in validation_generator:
        print(b.shape)


def _test_RBatchGenerator_PyTorch_filtered(tree_name: str, file_name: str, batch_size: int, chunk_size: int):
    print(f"Starting filtered PyTorch Test")
    filters = ["r_1 > 0.3", "r_2 < 0.8", "r_3 > 0.1"]

    train_generator, validation_generator = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
        tree_name, file_name, batch_size, chunk_size, filters=filters
    )

    for b in train_generator:
        print(b.shape)

    for b in validation_generator:
        print(b.shape)


def _test_RBatchGenerator_TensorFlow(tree_name: str, file_name: str, batch_size: int, chunk_size: int):
    print(f"Starting basic TensorFlow Test")
    train_ds, validation_ds = ROOT.TMVA.Experimental.CreateTFDatasets(tree_name, file_name, batch_size, chunk_size)

    for b in train_ds:
        print(b.shape)

    for b in validation_ds:
        print(b.shape)


def _test_RBatchGenerator_TensorFlow_filtered(tree_name: str, file_name: str, batch_size: int, chunk_size: int):
    print(f"Starting filtered TensorFlow Test")
    filters = ["r_1 > 0.3", "r_2 < 0.8", "r_3 > 0.1"]

    train_ds, validation_ds = ROOT.TMVA.Experimental.CreateTFDatasets(
        tree_name, file_name, batch_size, chunk_size, filters=filters
    )

    for b in train_ds:
        print(b.shape)

    for b in validation_ds:
        print(b.shape)


def _test_RBatchGenerator_vectors(tree_name: str, file_name: str, batch_size: int, chunk_size: int):
    max_vec_sizes = {"fB": 3, "fD": 3, "fF": 3, "fI": 3, "fL": 3, "fLL": 3, "fU": 3, "fUL": 3, "fULL": 3}

    train_generator, validation_generator = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        tree_name, file_name, batch_size, chunk_size, max_vec_sizes=max_vec_sizes, shuffle=False
    )

    for b in train_generator:
        print(b.shape)
        print(b)

        break

    for b in validation_generator:
        print(b.shape)
        break


class RBDT(unittest.TestCase):
    """
    Test RBDT interface
    """

    def test_RBatchGenerator_basic(self):
        _test_RBatchGenerator_NumPy()


if __name__ == "__main__":
    tree_name = "test_tree"
    file_name = "test_file.root"
    num_events = 10_000
    num_columns = 10

    batch_size = 1024
    chunk_size = 10_000

    createRDataFrame(tree_name, file_name, num_events, num_columns)
    _test_RBatchGenerator_NumPy(tree_name, file_name, batch_size, chunk_size)
    _test_RBatchGenerator_NumPy_filtered(tree_name, file_name, batch_size, chunk_size)
    _test_RBatchGenerator_PyTorch(tree_name, file_name, batch_size, chunk_size)
    _test_RBatchGenerator_PyTorch_filtered(tree_name, file_name, batch_size, chunk_size)
    _test_RBatchGenerator_TensorFlow(tree_name, file_name, batch_size, chunk_size)
    _test_RBatchGenerator_TensorFlow_filtered(tree_name, file_name, batch_size, chunk_size)

    print(f"Testing more types")
    _test_RBatchGenerator_vectors("test_tree", "../data/hvector.root", 5, 10)
    _test_RBatchGenerator_NumPy("test_tree", "../data/all_types.root", 5, 10)
