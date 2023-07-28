#include "iostream"
#include <vector>
#include <memory>

#include "TMVA/RChunkLoader.hxx"
#include "ROOT/RDataFrame.hxx"
#include "TMVA/RTensor.hxx"

void chunking_test()
{
    std::string fTreeName = "test_tree";
    std::string fFileName = "../data/small_data.root";
    std::vector<std::string> fCols = {"f1"};
    std::string fFilters = "f1 < 10";
    std::vector<size_t> fVecSizes = {};
    float fVecPadding = 0;

    size_t fChunkSize = 100;
    size_t fNumColumns = 1;
    size_t fCurrentRow = 0;

    std::unique_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor = std::unique_ptr<TMVA::Experimental::RTensor<float>>(
        new TMVA::Experimental::RTensor<float>({fChunkSize, fNumColumns}));

    // TMVA::Experimental::RChunkLoader<int &, float &, bool &>
    //     chunk_loader(fTreeName, fFileName, fChunkSize, fCols, fFilters, fVecSizes, fVecPadding);
    TMVA::Experimental::Internal::RChunkLoader<int &>
        chunk_loader(fTreeName, fFileName, fChunkSize, fCols, fFilters, fVecSizes, fVecPadding);

    std::pair res = chunk_loader.LoadChunk(*fChunkTensor, fCurrentRow);

    std::cout << "processed events: " << res.first << " passed events: " << res.second << std::endl;
}

int main()
{
    chunking_test();

    return 0;
}
