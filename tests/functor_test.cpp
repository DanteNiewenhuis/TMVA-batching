#include "iostream"
#include <vector>
#include <memory>

#include "TMVA/RChunkLoader.hxx"
#include "ROOT/RDataFrame.hxx"

#include "TMVA/RChunkLoader.hxx"

void chunking_test()
{
   std::string fTreeName = "test_tree";
   std::string fFileName = "../data/simple_data.root";
   std::vector<std::string> fCols = {"f1"};
   std::vector<size_t> fVecSizes = {};
   float fVecPadding = 0;

   size_t fChunkSize = 1;
   size_t fNumColumns = 1;
   size_t fCurrentRow = 0;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor = std::unique_ptr<TMVA::Experimental::RTensor<float>>(
       new TMVA::Experimental::RTensor<float>({fChunkSize, fNumColumns}));

   ROOT::RDataFrame x_rdf = ROOT::RDataFrame(fTreeName, fFileName);

   TMVA::Experimental::RChunkLoaderFunctor<int &> func(*fChunkTensor, fVecSizes, fVecPadding);

   x_rdf.Range(fChunkSize).Foreach(func, fCols);
}

int main()
{
   chunking_test();

   return 0;
}