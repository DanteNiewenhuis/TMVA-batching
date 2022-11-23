using namespace ROOT;
using namespace TMVA::Experimental;


class Generator_t {
    private:
        unsigned long i = 0;
    public:
        RTensor<float> operator() (const int& batch_size, RDataFrame& x_rdf) {
            RTensor<float> x_tensor = AsTensor<float>(x_rdf, x_rdf.GetColumnNames(), MemoryLayout::RowMajor);
            auto data_len = x_tensor.GetShape()[0];
            auto num_column = x_tensor.GetShape()[1];

            if(i+batch_size<data_len){
                RTensor<float> x_batch = x_tensor.Slice({{i,i+batch_size},{0,num_column}});
                i+=batch_size;
                return x_batch;
            }
            else{
                return x_tensor.Slice({{0,0},{0,0}});
            }
        }

        int get_i() {
            return i;
        }
};

void bg_slice_RDF_v1()
{
    RDataFrame x_rdf = RDataFrame("sig_tree", "Higgs_data.root", {"jet1_phi","jet1_eta", "jet2_pt"});
    Generator_t generator;
    int batch_size=4;
    bool batch_is_empty = 0;
    int batch_num = 0;

    while(!batch_is_empty)
        {
            auto batch = generator(batch_size,x_rdf);
            std::cout<<"Batch No. "<<batch_num<<std::endl;
            std::cout<<"Generator size: "<<batch.GetSize()<<std::endl;
            std::cout<<"Generator: "<<batch<<std::endl;
            batch_num++;
            if(batch.GetSize()==0) batch_is_empty = 1;
        }
}

int main(){
    bg_slice_RDF_v1();
}