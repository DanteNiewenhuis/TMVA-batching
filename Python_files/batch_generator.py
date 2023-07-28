import time
import ROOT
import numpy as np

main_folder = "../"


class BaseGenerator:
    def get_template(self, file_name, tree_name, columns=None, vec_sizes=None):
        """Generate a template for the BatchGenerator based on the given RDataFrame and columns

        Args:
            file_name (str): name of the root file.
            tree_name (str): name of the tree in the root file.
            columns (list[str]): Columns that should be loaded.
                                 Defaults to loading all Columns in the RDataFrame

        Returns:
            columns (list[str]): The columns of the DataFrame in the order that is used by the RDataFrame
            template (str): Template for the BatchGenerator
        """

        x_rdf = ROOT.RDataFrame(tree_name, file_name)

        if not columns:
            columns = x_rdf.GetColumnNames()

        template_dict = {
            "Int_t": "int&",
            "Float_t": "float&",
            "ROOT::VecOps::RVec<int>": "ROOT::RVec<int>",
            "ROOT::VecOps::RVec<float>": "ROOT::RVec<float>",
        }

        template_string = ""

        self.input_columns = []
        self.output_columns = []
        # Get the types of the different columns

        vec_size_idx = 0
        for name in columns:
            name_str = str(name)
            self.input_columns.append(name_str)
            column_type = template_dict[str(x_rdf.GetColumnType(name_str))]
            template_string += column_type + ","

            # If the column is a vector, add multiple columnnames to the output columns
            if column_type in ["ROOT::RVec<int>", "ROOT::RVec<float>"]:
                for i in range(vec_sizes[vec_size_idx]):
                    self.output_columns.append(f"{name_str}_{i}")
                vec_size_idx += 1

            else:
                self.output_columns.append(name_str)

        return template_string[:-1]

    def __init__(
        self,
        file_name,
        tree_name,
        chunk_size,
        batch_rows,
        columns=None,
        vec_sizes=None,
        filters=None,
        target=None,
        weights="",
        validation_split=1.0,
        max_chunks=0,
    ):
        """_summary_

        Args:
            file_name (str): name of the root file.
            tree_name (str): name of the tree in the root file.
            columns (list[str]): Columns that should be loaded.
            filters (list[str]): Filters that should be applied to the Data.
            chunk_size (int): The number of events in a chunk of data
            batch_rows (int): The number of events in a batch
            target (str, optional): The target that should be seperated from the rest of the data.
                                    If not given, all data is returned
        """

        # convert None types to lists for cppyy
        if vec_sizes == None:
            vec_sizes = []
        if columns == None:
            columns = []
        if filters == None:
            filters = []

        # TODO: better linking when importing into ROOT
        ROOT.gInterpreter.ProcessLine(
            f'#include "{main_folder}Cpp_files/BatchGenerator.cpp"'
        )

        template = self.get_template(file_name, tree_name, columns, vec_sizes)

        self.num_columns = len(self.output_columns)
        self.batch_rows = batch_rows
        self.batch_size = batch_rows * self.num_columns

        # Handle target
        self.target_given = target is not None
        if self.target_given:
            if target in self.output_columns:
                self.target_index = self.output_columns.index(target)
            else:
                raise ValueError(
                    f"Provided target not in given columns: \ntarget => {target}\ncolumns => {self.output_columns}"
                )

        # Handle weights
        self.weights_given = weights is not None
        if self.weights_given and not self.target_given:
            raise ValueError("Weights can only be used when a target is provided")
        if self.weights_given:
            if weights in self.output_columns:
                self.weights_index = self.output_columns.index(weights)
            else:
                raise ValueError(
                    f"Provided weights not in given columns: \nweights => {weights}\ncolumns => {self.output_columns}"
                )

        # Create C++ batch generator

        self.generator = ROOT.BatchGenerator(template)(
            file_name,
            tree_name,
            self.input_columns,
            filters,
            chunk_size,
            batch_rows,
            vec_sizes,
            validation_split,
            max_chunks,
            self.num_columns,
        )

        self.deactivated = False

    def start_validation(self):
        self.generator.start_validation()

    def Activate(self):
        """Initialize the generator to be used for a loop"""
        self.generator.init()

    def DeActivate(self):
        """Initialize the generator to be used for a loop"""
        self.generator.StopLoading()

    def GetSample(self):
        """Return a sample of data that has the same size and types as the actual result

        Returns:
            np.ndarray: data sample
        """
        if not self.target_given:
            return np.zeros((self.batch_rows, self.num_columns))

        if not self.weights_given:
            return np.zeros((self.batch_rows, self.num_columns - 1)), np.zeros(
                (self.batch_rows)
            )

        return (
            np.zeros((self.batch_rows, self.num_columns - 2)),
            np.zeros((self.batch_rows)),
            np.zeros((self.batch_rows)),
        )

    def BatchToNumpy(self, batch):
        """Convert a RTensor into a NumPy array

        Args:
            batch (RTensor): Batch returned from the BatchGenerator

        Returns:
            np.array: converted batch
        """
        data = batch.GetData()
        data.reshape((self.batch_size,))
        return_data = np.array(data).reshape(self.batch_rows, self.num_columns)

        # Splice target column from the data if weight is given
        if self.target_given:
            target_data = return_data[:, self.target_index]
            return_data = np.column_stack(
                (
                    return_data[:, : self.target_index],
                    return_data[:, self.target_index + 1 :],
                )
            )

            # Splice weights column from the data if weight is given
            if self.weights_given:
                if self.target_index < self.weights_index:
                    self.weights_index -= 1

                weights_data = return_data[:, self.weights_index]
                return_data = np.column_stack(
                    (
                        return_data[:, : self.weights_index],
                        return_data[:, self.weights_index + 1 :],
                    )
                )
                return return_data, target_data, weights_data

            return return_data, target_data

        return return_data

    def BatchToPyTorch(self, batch):
        """Convert a RTensor into a PyTorch tensor

        Args:
            batch (RTensor): Batch returned from the BatchGenerator

        Returns:
            torch.Tensor: converted batch
        """
        import torch

        data = batch.GetData()
        data.reshape((self.batch_size,))
        return_data = torch.Tensor(data).reshape(self.batch_rows, self.num_columns)

        # Splice target column from the data if weight is given
        if self.target_given:
            target_data = return_data[:, self.target_index]
            return_data = torch.column_stack(
                (
                    return_data[:, : self.target_index],
                    return_data[:, self.target_index + 1 :],
                )
            )

            # Splice weights column from the data if weight is given
            if self.weights_given:
                if self.target_index < self.weights_index:
                    self.weights_index -= 1

                weights_data = return_data[:, self.weights_index]
                return_data = torch.column_stack(
                    (
                        return_data[:, : self.weights_index],
                        return_data[:, self.weights_index + 1 :],
                    )
                )
                return return_data, target_data, weights_data

            return return_data, target_data

        return return_data

    def BatchToTF(self, batch):
        import tensorflow as tf

        batch = self.BatchToNumpy(batch)

        # TODO: improve this
        return batch

        if type(batch) == tuple:
            return [tf.constant(b, dtype=tf.float32) for b in batch]

        b = tf.constant(batch, dtype="float")

        return b

    # Return a batch when available
    def GetTrainBatch(self):
        """Return the next batch of data from the given RDataFrame

        Raises:
            StopIteration: Stop Iterating over data when all data has been processed

        Returns:
            X (np.ndarray): Batch of data of size.
            target (np.ndarray): Batch of Target data.
                                 Only given if a target column was specified
        """

        batch = self.generator.GetTrainBatch()

        if batch.GetSize() > 0:
            return batch

        return None

    def GetValidationBatch(self):
        """Return the next batch of data from the given RDataFrame

        Raises:
            StopIteration: Stop Iterating over data when all data has been processed

        Returns:
            X (np.ndarray): Batch of data of size.
            target (np.ndarray): Batch of Target data.
                                 Only given if a target column was specified
        """

        batch = self.generator.GetValidationBatch()

        if batch.GetSize() > 0:
            return batch

        return None


class TrainBatchGenerator:
    def __init__(self, base_generator: BaseGenerator, conversion_function):
        self.base_generator = base_generator
        self.conversion_function = conversion_function

    def Activate(self):
        self.base_generator.Activate()

    def DeActivate(self):
        self.base_generator.DeActivate()

    @property
    def columns(self):
        return self.base_generator.output_columns

    def __call__(self):
        self.Activate()

        while True:
            batch = self.base_generator.GetTrainBatch()

            if batch is None:
                break

            yield self.conversion_function(batch)


class ValidationBatchGenerator:
    def __init__(self, base_generator: BaseGenerator, conversion_function):
        self.base_generator = base_generator
        self.conversion_function = conversion_function

    @property
    def columns(self):
        return self.base_generator.output_columns

    def __call__(self):
        self.base_generator.start_validation()

        while True:
            batch = self.base_generator.GetValidationBatch()

            if batch is None:
                break

            yield self.conversion_function(batch)


def GetGenerators(
    file_name,
    tree_name,
    chunk_size,
    batch_rows,
    columns=None,
    vec_sizes=None,
    filters=[],
    target=None,
    weights=None,
    validation_split=0.1,
    max_chunks=1,
):
    base_generator = BaseGenerator(
        file_name,
        tree_name,
        chunk_size,
        batch_rows,
        columns,
        vec_sizes,
        filters,
        target,
        weights,
        validation_split,
        max_chunks,
    )

    train_generator = TrainBatchGenerator(base_generator, base_generator.BatchToNumpy)
    validation_generator = ValidationBatchGenerator(
        base_generator, base_generator.BatchToNumpy
    )

    return train_generator, validation_generator


def CreateTFDatasets(
    file_name,
    tree_name,
    chunk_size,
    batch_rows,
    columns=None,
    vec_sizes=None,
    filters=[],
    target=None,
    weights=None,
    validation_split=0.1,
    max_chunks=0,
):
    import tensorflow as tf

    base_generator = BaseGenerator(
        file_name,
        tree_name,
        chunk_size,
        batch_rows,
        columns,
        vec_sizes,
        filters,
        target,
        weights,
        validation_split,
        max_chunks,
    )

    train_generator = TrainBatchGenerator(base_generator, base_generator.BatchToTF)
    validation_generator = ValidationBatchGenerator(
        base_generator, base_generator.BatchToTF
    )

    num_columns = len(train_generator.columns)

    # No target and weights given
    if target == None:
        batch_signature = tf.TensorSpec(
            shape=(batch_rows, num_columns), dtype=tf.float32
        )

    # Target given, no weights given
    if target != None and weights == None:
        batch_signature = (
            tf.TensorSpec(shape=(batch_rows, num_columns - 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_rows,), dtype=tf.float32),
        )

    # Target given, no weights given
    if target != None and weights != None:
        batch_signature = (
            tf.TensorSpec(shape=(batch_rows, num_columns - 2), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_rows,), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_rows,), dtype=tf.float32),
        )

    ## TODO: Add support for no target en weights
    ds_train = tf.data.Dataset.from_generator(
        train_generator, output_signature=batch_signature
    )

    ds_validation = tf.data.Dataset.from_generator(
        validation_generator, output_signature=batch_signature
    )

    return ds_train, ds_validation


def GetPyTorchDataLoader(
    file_name,
    tree_name,
    chunk_size,
    batch_rows,
    columns=None,
    vec_sizes=None,
    filters=[],
    target=None,
    weights=None,
    validation_split=0.1,
    max_chunks=0,
):
    base_generator = BaseGenerator(
        file_name,
        tree_name,
        chunk_size,
        batch_rows,
        columns,
        vec_sizes,
        filters,
        target,
        weights,
        validation_split,
        max_chunks,
    )

    train_generator = TrainBatchGenerator(base_generator, base_generator.BatchToPyTorch)
    validation_generator = ValidationBatchGenerator(
        base_generator, base_generator.BatchToPyTorch
    )

    return train_generator, validation_generator
