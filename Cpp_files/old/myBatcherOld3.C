#include <string>
#include <vector>
#include <iostream>
#include <utility>

using namespace ROOT;
using namespace TMVA::Experimental;

template<typename T> 
class HelperFunc {
private: 
    size_t offset = 0, num_columns, offset_threshold, i;
    TMVA::Experimental::RTensor<float>* x_tensor;

public:
    HelperFunc(RTensor<float>* x_tensor, const size_t num_columns, const size_t data_len)
        : x_tensor(x_tensor), num_columns(num_columns) 
    {
        offset_threshold = data_len * num_columns;    
    }

    // template <typename First_T> 
    // void assign_to_tensor(First_T first) {
    //     x_tensor->GetData()[offset + i] = first;

    // }

    // template <typename First_T, typename... Rest_T> 
    // void assign_to_tensor(First_T first, Rest_T... rest) {
    //     x_tensor->GetData()[offset + i] = first;

    //     i += 1;

    //     assign_to_tensor(std::forward<Rest_T>(rest)...);
    // }

    // TODO: add a data_len to the class
    template <typename... Args>
    void operator()(Args... args) 
    {
        // if (offset >= offset_threshold) return;

        // i = 0;

        ((std::cout << args << ' '), ...);

        std::cout << std::endl;

        // offset += num_columns;
    }


};

class Generator_t {
private:
    size_t i = 0, batch_size, data_len, num_columns;
    RTensor<float>* x_tensor;

public:

    Generator_t (RDataFrame &x_rdf, std::vector<std::string> cols, 
                    const size_t batch_size, const size_t data_len): 
                    batch_size(batch_size), data_len(data_len), num_columns(cols.size()) 
    {
        // Create Chunk
        x_tensor = new RTensor<float>({data_len, num_columns});

        // Fill Chunk
        // HelperFunc<float&, float&, float&, float&> func(x_tensor, num_columns, data_len);
        HelperFunc<float&> func(x_tensor, num_columns, data_len);

        func(1, 2, 3, 5, 6);

        x_rdf.Foreach(func, cols);
    }   

    RTensor<float> operator()() 
    {
        if (i * batch_size + batch_size <= data_len) {
            unsigned long offset = i * batch_size * num_columns;
            RTensor<float> x_batch(x_tensor->GetData() + offset, {batch_size, num_columns});

            i++;
            return x_batch;
        }
        else {
            // TODO: create option for returning the final rows as a smaller batch
            return x_tensor->Slice({{0, 0}, {0, 0}});
        }
    }

    int get_batch_size() {return batch_size; }

};



void myBatcher3() {
    // define variables
    std::vector<std::string> cols = { "m_jj", "m_jjj", "m_jlv", "m_lv"};
    int batch_size = 4, data_len = 10;

    RDataFrame x_rdf =
      RDataFrame("sig_tree", "Higgs_data.root", cols);

    auto column_names = x_rdf.GetColumnNames();

    // std::cout << column_names << std::endl;

    // define generator
    Generator_t generator(x_rdf, cols, batch_size, data_len);

    // generate batches
    // while (true) {
    //     auto batch = generator();

    //     std::cout << "Batch: " << batch << std::endl;

    //     if (batch.GetSize() == 0) {
    //         break;
    //     }
    // }

}

int main() {
    myBatcher3();
}