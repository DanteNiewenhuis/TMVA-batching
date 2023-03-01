import time
import ROOT
import numpy as np

main_folder = "../"


class BaseGenerator:

    def get_template(self, file_name: str, tree_name: str, columns: list[str] = None) -> tuple[list[str], str]:
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

        if columns:
            x_rdf = ROOT.RDataFrame(tree_name, file_name, columns)
        else:
            # Load all columns if no columns are given
            x_rdf = ROOT.RDataFrame(tree_name, file_name)

        template_dict = {"Int_t": "int&", "Float_t": "float&", 
                         "ROOT::VecOps::RVec<int>": "ROOT::RVec<int>", 
                         "ROOT::VecOps::RVec<float>": "ROOT::RVec<float>"}

        template_string = ""
        self.columns = list(x_rdf.GetColumnNames())

        # Get the types of the different columns
        for name in self.columns:
            template_string += template_dict[x_rdf.GetColumnType(name)] + ","

        return template_string[:-1]

    def __init__(self, file_name: str, tree_name: str, chunk_rows: int, batch_rows: int,
                 columns: list[str] = None, vec_sizes: list[int] = None, filters: list[str] = None, target: str = None, 
                 weights: str = "", validation_split: float = 1.0, max_chunks: int = 0,
                 output_type: str = "NumPy"):
        """_summary_

        Args:
            file_name (str): name of the root file.
            tree_name (str): name of the tree in the root file.
            columns (list[str]): Columns that should be loaded.
            filters (list[str]): Filters that should be applied to the Data.
            chunk_rows (int): The number of events in a chunk of data
            batch_rows (int): The number of events in a batch
            target (str, optional): The target that should be seperated from the rest of the data.
                                    If not given, all data is returned 
        """


        # TODO: better linking when importing into ROOT
        ROOT.gInterpreter.ProcessLine(
            f'#include "{main_folder}Cpp_files/BatchGenerator.cpp"')

        template = self.get_template(file_name, tree_name, columns)

        if vec_sizes is None: vec_sizes = []
        if len(vec_sizes) != template.count("RVec"):
            raise ValueError (
                f"Incorrect number of vector sizes provided. \nSizes given: {vec_sizes} for {template.count('RVec')} vectors"
            )

        self.num_columns = len(self.columns) + sum(vec_sizes) - len(vec_sizes)
        self.batch_rows = batch_rows
        self.batch_size = batch_rows * self.num_columns

        # Handle target
        self.target_given = target is not None
        if self.target_given:
            if target in self.columns:
                self.target_index = self.columns.index(target)
            else:
                raise ValueError(
                    f"Provided target not in given columns: \ntarget => {target}\ncolumns => {self.columns}")

        # Handle weights
        self.weights_given = weights is not None
        if self.weights_given and not self.target_given:
            raise ValueError(
                "Weights can only be used when a target is provided")
        if self.weights_given:
            if weights in self.columns:
                self.weights_index = self.columns.index(weights)
            else:
                raise ValueError(
                    f"Provided weights not in given columns: \nweights => {weights}\ncolumns => {self.columns}")

        # Create C++ batch generator

        print(f"{template = }")

        self.generator = ROOT.BatchGenerator(template)(
            file_name, tree_name, self.columns, filters, chunk_rows, batch_rows, vec_sizes, validation_split, max_chunks, self.num_columns)

        self.deactivated = False
    
    def Activate(self):
        """Initialize the generator to be used for a loop
        """
        self.generator.init()

    def DeActivate(self):
        """Initialize the generator to be used for a loop
        """
        self.generator.StopLoading()


    def GetSample(self):
        if not self.target_given:
            return np.zeros((self.batch_rows, self.num_columns))

        if not self.weights_given:
            return np.zeros((self.batch_rows, self.num_columns-1)), np.zeros((self.batch_rows))

        return np.zeros((self.batch_rows, self.num_columns-2)), np.zeros((self.batch_rows)), np.zeros((self.batch_rows))

    def BatchToNumpy(self, batch):
        data = batch.GetData()
        data.reshape((self.batch_size,))
        return_data = np.array(data).reshape(
            self.batch_rows, self.num_columns)

        # Splice target column from the data if weight is given
        if self.target_given:
            target_data = return_data[:, self.target_index]
            return_data = np.column_stack(
                (return_data[:, :self.target_index], return_data[:, self.target_index+1:]))

            # Splice weights column from the data if weight is given
            if self.weights_given:
                if self.target_index < self.weights_index:
                    self.weights_index -= 1

                weights_data = return_data[:, self.weights_index]
                return_data = np.column_stack(
                    (return_data[:, :self.weights_index], return_data[:, self.weights_index+1:]))
                return return_data, target_data, weights_data

            return return_data, target_data

        return return_data

    def BatchToPyTorch(self, batch):
        import torch

        data = batch.GetData()
        data.reshape((self.batch_size,))
        return_data = torch.Tensor(data).reshape(
            self.batch_rows, self.num_columns)

        # Splice target column from the data if weight is given
        if self.target_given:
            target_data = return_data[:, self.target_index]
            return_data = torch.column_stack(
                (return_data[:, :self.target_index], return_data[:, self.target_index+1:]))

            # Splice weights column from the data if weight is given
            if self.weights_given:
                if self.target_index < self.weights_index:
                    self.weights_index -= 1

                weights_data = return_data[:, self.weights_index]
                return_data = torch.column_stack(
                    (return_data[:, :self.weights_index], return_data[:, self.weights_index+1:]))
                return return_data, target_data, weights_data

            return return_data, target_data

        return return_data
    
    def BatchToTF(self, batch):
        import tensorflow as tf

        batch = self.BatchToNumpy(batch)

        # if type(batch) == tuple:
        #     return [tf.constant(b, dtype=tf.float32) for b in batch] 

        b = tf.constant(batch, dtype="float")

        print(f"{type(b) = }, {b = }")
        return b
    
    # Return a batch when available
    def GetTrainBatch(self) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Return the next batch of data from the given RDataFrame

        Raises:
            StopIteration: Stop Iterating over data when all data has been processed

        Returns:
            X (np.ndarray): Batch of data of size.
            target (np.ndarray): Batch of Target data. 
                                 Only given if a target column was specified
        """

        batch = self.generator.GetTrainBatch()

        if (batch.GetSize() > 0):
            return batch

        return None

    def GetValidationBatch(self) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Return the next batch of data from the given RDataFrame

        Raises:
            StopIteration: Stop Iterating over data when all data has been processed

        Returns:
            X (np.ndarray): Batch of data of size.
            target (np.ndarray): Batch of Target data. 
                                 Only given if a target column was specified
        """

        batch = self.generator.GetValidationBatch()

        if (batch.GetSize() > 0):
            return batch

        return None

class TrainBatchGenerator:

    def __init__(self, base_generator: BaseGenerator, conversion_function):
        self.base_generator = base_generator
        self.conversion_function = conversion_function
    
    def Activate(self):
        print("TrainBatchGenerator => Activate")
        self.base_generator.Activate()
    
    def DeActivate(self):
        self.base_generator.DeActivate()
    
    @property
    def columns(self) -> list[str]:
        return self.base_generator.columns

    def __call__(self):
        self.Activate()

        while(True):
            batch = self.base_generator.GetTrainBatch()

            if batch is None:
                break
            
            yield self.conversion_function(batch)

class ValidationBatchGenerator:

    def __init__(self, base_generator: BaseGenerator, conversion_function):
        self.base_generator = base_generator
        self.conversion_function = conversion_function

    @property
    def columns(self) -> list[str]:
        return self.base_generator.columns

    def __call__(self):
        while(True):
            batch = self.base_generator.GetValidationBatch()

            if batch is None:
                break
            
            yield self.conversion_function(batch)


def GetGenerators(file_name: str, tree_name: str, chunk_rows: int, batch_rows: int,
                 columns: list[str] = None, vec_sizes = None, filters: list[str] = [], target: str = None, 
                 weights: str = None, validation_split: float = 0, max_chunks: int = 1):
    
    base_generator = BaseGenerator(file_name, tree_name, chunk_rows, batch_rows,
                 columns, vec_sizes, filters, target, weights, validation_split, max_chunks)

    train_generator = TrainBatchGenerator(base_generator, base_generator.BatchToNumpy)
    validation_generator = ValidationBatchGenerator(base_generator, base_generator.BatchToNumpy)

    return train_generator, validation_generator

def GetTFDatasets(file_name: str, tree_name: str, chunk_rows: int, batch_rows: int,
                 columns: list[str] = None, vec_sizes: list[int] = None, filters: list[str] = [], target: str = None, 
                 weights: str = None, validation_split: float = 0, max_chunks: int = 0):

    import tensorflow as tf

    base_generator = BaseGenerator(file_name, tree_name, chunk_rows, batch_rows,
                 columns, vec_sizes, filters, target, weights, validation_split, max_chunks)

    train_generator = TrainBatchGenerator(base_generator, base_generator.BatchToTF)
    validation_generator = ValidationBatchGenerator(base_generator, base_generator.BatchToTF)

    num_columns = len(train_generator.columns)

    print(f"{num_columns = }")

    # No target and weights given
    if (target == None):
        batch_signature = (tf.TensorSpec(shape=(batch_rows ,num_columns), dtype=tf.float32))

    # Target given, no weights given
    if (target != None and weights == None):
        batch_signature = ( tf.TensorSpec(shape=(batch_rows, num_columns-1), dtype=tf.float32), 
                            tf.TensorSpec(shape=(batch_rows,), dtype=tf.float32))

    # Target given, no weights given
    if (target != None and weights != None):
        batch_signature = ( tf.TensorSpec(shape=(batch_rows, num_columns-2), dtype=tf.float32), 
                            tf.TensorSpec(shape=(batch_rows,), dtype=tf.float32),
                            tf.TensorSpec(shape=(batch_rows,), dtype=tf.float32))

    ## TODO: Add support for no target en weights
    ds_train = tf.data.Dataset.from_generator(train_generator, output_signature = batch_signature)

    ds_validation = tf.data.Dataset.from_generator(validation_generator, output_signature = batch_signature)


    return ds_train, ds_validation

def GetPyTorchDataLoader(file_name: str, tree_name: str, chunk_rows: int, batch_rows: int,
                 columns: list[str] = None, vec_sizes: list[int] = None, filters: list[str] = [], target: str = None, 
                 weights: str = None, validation_split: float = 0, max_chunks: int = 0):

    base_generator = BaseGenerator(file_name, tree_name, chunk_rows, batch_rows,
                 columns, vec_sizes, filters, target, weights, validation_split, max_chunks)

    train_generator = TrainBatchGenerator(base_generator, base_generator.BatchToPyTorch)
    validation_generator = ValidationBatchGenerator(base_generator, base_generator.BatchToPyTorch)

    return train_generator, validation_generator
