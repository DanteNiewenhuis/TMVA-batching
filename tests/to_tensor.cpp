#include "iostream"
#include <vector>
#include <memory>

#include "TMVA/RChunkLoader.hxx"
#include "ROOT/RDataFrame.hxx"

template <typename... Args>
std::unique_ptr<TMVA::Experimental::RTensor<float>> to_tensor(ROOT::RDataFrame &x_rdf)
{
   std::vector<std::string> fCols = x_rdf.GetColumnNames();
   size_t fNumColumns = fCols.size();
   auto myCount = x_rdf.Count();
   size_t fChunkSize = myCount.GetValue();

   auto fChunkTensor = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fChunkSize, fNumColumns});

   // auto fChunkTensor = TMVA::Experimental::RTensor<float>({fChunkSize, fNumColumns});
   TMVA::Experimental::RChunkLoaderFunctor<Args...> func(*fChunkTensor);

   x_rdf.Foreach(func, fCols);

   return std::move(fChunkTensor);
}

int main()
{
}

// std::unique_ptr<TMVA::Experimental::RTensor<float>> to_tensor_wrapper(ROOT::RDataFrame &x_rdf)
// {
//    return to_tensor<int &, float &, bool &>(x_rdf);
// }

// int main()
// {

//    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("test_tree", "../data/simple_data.root");
//    auto x_tensor = to_tensor<int &, float &, bool &>(x_rdf);

//    std::cout << "x_tensor: " << x_tensor->GetData()[0] << std::endl;
//    std::cout << "x_tensor: " << x_tensor->GetData()[3] << std::endl;
//    std::cout << "x_tensor: " << x_tensor->GetData()[5] << std::endl;

//    return 0;
// }