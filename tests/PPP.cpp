#include "iostream"
#include <vector>
#include <memory>

#include "TMVA/RChunkLoader.hxx"
#include "ROOT/RDataFrame.hxx"
#include "TMVA/RTensor.hxx"

void chunking_test()
{
    std::string fTreeName = "test_tree";
    std::string fFileName = "../data/simple_data.root";
    std::vector<std::string> fCols = {"f1"};
    std::vector<std::string> fFilters = {};
    std::vector<size_t> fVecSizes = {};
    float fVecPadding = 0;

    size_t fChunkSize = 1;
    size_t fNumColumns = 1;
    size_t fCurrentRow = 0;

    TMVA::Experimental::RTensor<float> fChunkTensor =
        TMVA::Experimental::RTensor<float>({fChunkSize, fNumColumns});

    TMVA::Experimental::RChunkLoader<int &, float &, bool &>
        chunk_loader(fTreeName, fFileName, fChunkSize, fCols, fFilters, fVecSizes, fVecPadding);

    size_t processed_events = chunk_loader.LoadChunk(*fChunkTensor, fCurrentRow);
}

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

    TMVA::Experimental::RBatchGenerator<int &, float &, bool &>
        generator(fTreeName, fFileName, fChunkSize, fBatchSize, fCols);

    generator.Activate();
    TMVA::Experimental::RTensor<float> *batch;
    while (generator.HasTrainData())
    {
        batch = generator.GetTrainBatch();
        // Processing the batch
    }
    generator.DeActivate();

    generator.StartValidation();
    while (generator.HasValidationData())
    {
        batch = generator.GetValidationBatch();
        // Processing the batch
    }
}

void batching_test()
{
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
        // Process Training batch
    }

    fBatchLoader.StartValidation();
    while (fBatchLoader.HasValidationData())
    {
        current_batch = fBatchLoader.GetValidationBatch();
        // Process Training batch
    }
}

int main()
{
    chunking_test();

    return 0;
}
