#include "iostream"
#include <vector>
#include <memory>

#include "TMVA/RBatchGenerator.hxx"
#include "ROOT/RDataFrame.hxx"

void generator_test()
{
   std::string fTreeName = "test_tree";
   std::string fFileName = "../data/Higgs_data_full.root";
   ROOT::RDataFrame x_rdf = ROOT::RDataFrame(fTreeName, fFileName);

   std::vector<std::string> fCols = x_rdf.GetColumnNames();

   // std::vector<std::string> fCols = {"f1"};
   std::vector<std::string> fFilters = {};
   std::vector<size_t> fVecSizes = {};
   float fVecPadding = 0, fValidationSplit = 0.3;

   size_t fChunkSize = 10000;
   size_t fBatchSize = 1000;
   size_t fNumColumns = fCols.size();
   size_t fCurrentRow = 0;

   // TMVA::Experimental::RBatchGenerator<int &, float &, bool &> generator(fTreeName, fFileName, fChunkSize, fBatchSize, fCols);
   TMVA::Experimental::RBatchGenerator<float &, float &, float &, float &, float &,
                                       float &, float &, float &, float &, float &,
                                       float &, float &, float &, float &, float &,
                                       float &, float &, float &, float &, float &,
                                       float &, float &, float &, float &, float &,
                                       float &, float &, float &, float &>
       generator(fTreeName, fFileName, fChunkSize, fBatchSize, fCols, fFilters, fVecSizes, fVecPadding, fValidationSplit);

   generator.Activate();

   TMVA::Experimental::RTensor<float> *batch;
   while (generator.HasTrainData())
   {
      batch = generator.GetTrainBatch();
      std::cout << "training batch: " << batch << std::endl;
      std::cout << "training batch size: " << batch->GetSize() << std::endl;
      break;
      throw std::runtime_error("");
   }
   generator.DeActivate();

   generator.StartValidation();

   while (generator.HasValidationData())
   {
      batch = generator.GetValidationBatch();
      std::cout << "validation batch: " << batch << std::endl;
      std::cout << "validation batch size: " << batch->GetSize() << std::endl;
      break;
   }

   std::cout << "End of File" << std::endl;
}

int main()
{
   generator_test();
}