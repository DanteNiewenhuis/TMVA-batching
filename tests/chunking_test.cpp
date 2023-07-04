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

    std::unique_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor = std::unique_ptr<TMVA::Experimental::RTensor<float>>(
        new TMVA::Experimental::RTensor<float>({fChunkSize, fNumColumns}));

    // TMVA::Experimental::RChunkLoader<int &, float &, bool &>
    //     chunk_loader(fTreeName, fFileName, fChunkSize, fCols, fFilters, fVecSizes, fVecPadding);
    TMVA::Experimental::RChunkLoader<int &>
        chunk_loader(fTreeName, fFileName, fChunkSize, fCols, fFilters, fVecSizes, fVecPadding);

    size_t processed_events = chunk_loader.LoadChunk(*fChunkTensor, fCurrentRow);
}

int main()
{
    chunking_test();

    return 0;
}