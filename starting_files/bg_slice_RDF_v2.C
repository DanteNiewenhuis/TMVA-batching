using namespace ROOT;
using namespace TMVA::Experimental;

class Generator_t {
    private:
        unsigned long i = 0;
    public:
        RTensor<float> operator() (const unsigned long& batch_size, RDataFrame& x_rdf) {
            RTensor<float> x_tensor = AsTensor<float>(x_rdf, x_rdf.GetColumnNames(), MemoryLayout::RowMajor);
            auto data_len = x_tensor.GetShape()[0];
            auto num_column = x_tensor.GetShape()[1];


            if(i+batch_size<data_len){
                unsigned long offset = i * batch_size * num_column;
                RTensor<float> x_batch(x_tensor.GetData()+offset, {batch_size,num_column} );
                // RTensor<float> x_batch = x_tensor.Slice({{i,i+batch_size},{0,num_column}});
                i+=batch_size;
                return x_batch;
            }
            else{
                return x_tensor.Slice({{0,0},{0,0}});
            }
    }
};

void bg_slice_RDF()
{
    RDataFrame x_rdf = RDataFrame("sig_tree", "Higgs_data.root", {"jet1_phi","jet1_eta", "jet2_pt"});
    Generator_t generator;
    int batch_size=4;
    bool batch_is_empty = 0;
    int batch_num = 0;
    while(!batch_is_empty)
        {
            auto batch = generator(batch_size,x_rdf);
            cout<<"Batch No. "<<batch_num<<endl;
            cout<<"Generator: "<<batch<<endl;
            batch_num++;
            if(batch.GetSize()==0) batch_is_empty = 1;
        }
}

int main(){
    bg_slice_RDF();
}