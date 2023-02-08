import ROOT
import numpy as np

main_folder = "../"


class BatchGenerator:

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

        template_dict = {"Int_t": "int&", "Float_t": "float&"}

        template_string = ""
        columns = list(x_rdf.GetColumnNames())

        # Get the types of the different columns
        for name in columns:
            template_string += template_dict[x_rdf.GetColumnType(name)] + ","

        return columns, template_string[:-1]

    def __init__(self, file_name: str, tree_name: str, chunk_rows: int, batch_rows: int,
                 columns: list[str] = None, filters: list[str] = [], target: str = None, weights: str = None):
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

        columns, template = self.get_template(file_name, tree_name, columns)

        self.batch_rows = batch_rows
        self.num_columns = len(columns)
        self.batch_size = batch_rows * self.num_columns

        # Handle target
        self.target_given = target is not None
        if self.target_given:
            if target in columns:
                self.target_index = columns.index(target)
            else:
                raise ValueError(
                    f"Provided target not in given columns: \ntarget => {target}\ncolumns => {columns}")

        # Handle weights
        self.weights_given = weights is not None
        if self.weights_given and not self.target_given:
            raise ValueError(
                "Weights can only be used when a target is provided")
        if self.weights_given:
            if weights in columns:
                self.weights_index = columns.index(weights)
            else:
                raise ValueError(
                    f"Provided weights not in given columns: \nweights => {weights}\ncolumns => {columns}")

        # Create C++ batch generator
        self.generator = ROOT.BatchGenerator(template)(
            file_name, tree_name, columns, filters, chunk_rows, batch_rows)

    def __iter__(self):
        """Initialize the generator to be used for a loop
        """
        self.generator.init()

        return self

    # Return a batch when available
    def __next__(self) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Return the next batch of data from the given RDataFrame

        Raises:
            StopIteration: Stop Iterating over data when all data has been processed

        Returns:
            X (np.ndarray): Batch of data of size.
            target (np.ndarray): Batch of Target data. 
                                 Only given if a target column was specified
        """

        batch = self.generator.GetBatch()

        if (batch.GetSize() > 0):
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

            else:
                return_data

        raise StopIteration
