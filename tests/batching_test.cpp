#include "iostream"
#include <vector>
#include <memory>

#include "TMVA/RTensor.hxx"
#include "TMVA/RBatchLoader.hxx"

void batching_test()
{
   const size_t fBatchSize = 10, fChunkSize = 100, fNumColumns = 10, fMaxBatches = 10, fNumBatches = fChunkSize / fBatchSize;
   const float fValidationSplit = 0.3;

   TMVA::Experimental::RTensor<float> x_tensor = TMVA::Experimental::RTensor<float>({fChunkSize, fNumColumns});
   TMVA::Experimental::RBatchLoader fBatchLoader(fBatchSize, fNumColumns, fMaxBatches);

   // Create indices
   std::vector<size_t> row_order = std::vector<size_t>(fChunkSize);
   size_t num_validation = ceil(fChunkSize * fValidationSplit);

   std::vector<size_t> valid_idx({row_order.begin(), row_order.begin() + num_validation});
   std::vector<size_t> train_idx({row_order.begin() + num_validation, row_order.end()});

   // Create Batches
   fBatchLoader.Activate(); // Activate the loading
   fBatchLoader.CreateTrainingBatches(x_tensor, train_idx);
   fBatchLoader.CreateValidationBatches(x_tensor, valid_idx);
   fBatchLoader.DeActivate(); // DeActivate the Loading after the batches are created

   TMVA::Experimental::RTensor<float> *current_batch;
   while (fBatchLoader.HasTrainData())
   {
      current_batch = fBatchLoader.GetTrainBatch();
      std::cout << "training batch: " << current_batch->GetSize() << std::endl;
   }

   fBatchLoader.StartValidation();
   while (fBatchLoader.HasValidationData())
   {
      current_batch = fBatchLoader.GetValidationBatch();
      std::cout << "validation batch: " << current_batch->GetSize() << std::endl;
   }

   std::cout << "Epoch 2" << std::endl;
   fBatchLoader.Activate(); // Activate the loading
   fBatchLoader.CreateTrainingBatches(x_tensor, train_idx);
   fBatchLoader.DeActivate(); // DeActivate the Loading after the batches are created
   while (fBatchLoader.HasTrainData())
   {
      current_batch = fBatchLoader.GetTrainBatch();
      std::cout << "training batch: " << current_batch->GetSize() << std::endl;
   }

   fBatchLoader.StartValidation();
   while (fBatchLoader.HasValidationData())
   {
      current_batch = fBatchLoader.GetValidationBatch();
      std::cout << "validation batch: " << current_batch->GetSize() << std::endl;
   }

   std::cout << "EndOfFile" << std::endl;
}

int main()
{
   batching_test();

   return 0;
}