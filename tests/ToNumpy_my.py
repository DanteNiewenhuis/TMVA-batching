import time
import numpy as np
import ROOT

start = time.perf_counter()

end = time.perf_counter()

print(f"jitting took {end - start}")


def get_template(x_rdf, cols):
    template_dict = {
        "Int_t": "int&",
        "Float_t": "float&",
        "Bool_t": "bool&",
        "ROOT::VecOps::RVec<int>": "ROOT::RVec<int>",
        "ROOT::VecOps::RVec<float>": "ROOT::RVec<float>",
        "ROOT::VecOps::RVec<bool>": "ROOT::RVec<bool>",
    }

    template_string = ""

    for name in cols:
        name_str = str(name)
        column_type = template_dict[str(x_rdf.GetColumnType(name_str))]
        template_string += column_type + ","

    return template_string[:-1]


def ToNumpy(x_rdf):
    cols = x_rdf.GetColumnNames()
    num_columns = cols.size()

    template = get_template(x_rdf, cols)

    ROOT.gInterpreter.ProcessLine('#include "to_tensor.cpp"')
    print(f"{template = }")

    s = """
std::unique_ptr<TMVA::Experimental::RTensor<float>> to_tensor_wrapper(ROOT::RDataFrame &x_rdf)
{
    auto x_tensor = to_tensor<"""

    s += template
    s += """>(x_rdf);
    return std::move(x_tensor);
}
    """

    print(s)
    ROOT.gInterpreter.Declare(s)

    x_tensor = ROOT.to_tensor_wrapper(x_rdf)

    rows = int(x_tensor.GetSize() / num_columns)

    data = x_tensor.GetData()
    data.reshape((rows * num_columns,))

    return np.array(data).reshape(rows, num_columns)


start = time.perf_counter()
tree_name = "test_tree"
file_name = "../data/simple_data.root"

x_rdf = ROOT.RDataFrame(tree_name, file_name)
end = time.perf_counter()

print(f"Getting rdf took {end - start}")

start = time.perf_counter()
array = ToNumpy(x_rdf)
end = time.perf_counter()

print(f"ToNumpy took {end - start}")

print(f"{array}")
