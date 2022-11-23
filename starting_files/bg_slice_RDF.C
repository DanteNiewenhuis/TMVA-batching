#include <cstddef>
#include <string>
#include <vector>

using namespace ROOT;
using namespace TMVA::Experimental;

// Make a template for the possible data. triple dots means that this is done for all following arguments
template <size_t NCols, typename First, typename... Rest> 
class HelperFunc {
      static_assert(1 + sizeof...(Rest) == NCols, ""); // check if the proper number of columns are given

private:
  // For N = 0, 1, ..., NCols - 1
  template <size_t N> struct AssignToTensor {
    template <typename FirstArgs, typename... RestArgs>
    static void Call(TMVA::Experimental::RTensor<float> &x,
                     const size_t offset,
                     FirstArgs &&first, RestArgs &&... rest) {
      // Assign x[offset + N] = first
      x.GetData()[offset + N] = first; // Set the first 
      // Assign x[offset + N - 1] = first element of rest...
      AssignToTensor<N + 1>::Call(x, offset, std::forward<RestArgs>(rest)...);
    }
  };
  // Stop at N = NCols, do nothing
  template <> struct AssignToTensor<NCols> {
    template <typename... Args>
    static void Call(TMVA::Experimental::RTensor<float> &, const size_t offset,
                     Args...) {}
  };

private:
  size_t offset = 0;
  TMVA::Experimental::RTensor<float> &fTensor;

public:
  HelperFunc(TMVA::Experimental::RTensor<float> &x_tensor)
      : fTensor(x_tensor) {}

  void operator()(First first, Rest... rest) {
    AssignToTensor<0>::Call(fTensor, offset, std::forward<First>(first), std::forward<Rest>(rest)...);
    //offset += NCols;
  };

};

class Generator_t {
private:
  size_t i = 0;

public:
  RTensor<float> operator()(const size_t batch_size, RDataFrame &x_rdf,
                            const size_t nevt) {

    std::vector<std::string> cols = { "m_jj", "m_jjj", "m_jlv", "m_lv"};
    size_t numCols = cols.size();


    TMVA::Experimental::RTensor<float> x_tensor({nevt, numCols});
    HelperFunc<4, float&, float&, float&, float&> func(x_tensor);
    size_t offset = 0;

    //TO DO: make column input dynamic with std::make_index::sequence
    //x_rdf.GetColumnNames(); // take an array of column names

    x_rdf.Foreach(func, cols);

    auto data_len = x_tensor.GetShape()[0];
    auto num_column = x_tensor.GetShape()[1];
    std::cout << "data len = " << data_len << " and num_column = " << num_column
              << std::endl;
    // std::cout << "Rtensor = \n";
    // std::cout << x_tensor << std::endl;

    if (i + batch_size < data_len) {
      unsigned long offset = i * batch_size * num_column;
      RTensor<float> x_batch(x_tensor.GetData() + offset,
                             {batch_size, num_column});

      i += batch_size;
      return x_batch;
    }
    else {
      return x_tensor.Slice({{0, 0}, {0, 0}});
    }
  }
};

void bg_slice_RDF() {
  RDataFrame x_rdf =
      RDataFrame("sig_tree", "Higgs_data.root", {"jet1_phi", "jet1_eta", "jet2_pt"});

  // auto c = x_rdf.Count();
  // cout << "dataframe size: " << c << endl;

  Generator_t generator;
  int batch_size = 1;
  bool batch_is_empty = 0;
  int batch_num = 0;
  size_t nevt = 2; // chuck size
  while (!batch_is_empty) {
    auto batch = generator(batch_size, x_rdf, nevt);
    cout << "Batch No. " << batch_num << endl;
    cout << "Generator: " << batch << endl;
    break;
    batch_num++;
    if (batch.GetSize() == 0)
      batch_is_empty = 1;
  }
}

int main() { bg_slice_RDF(); }