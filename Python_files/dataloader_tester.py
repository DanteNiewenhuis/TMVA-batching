import ROOT

main_folder = "../"

ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/ChunkLoader.cpp"')

ROOT.gInterpreter.ProcessLine("""
size_t load_chunk(TMVA::Experimental::RTensor<float>& x_tensor, ROOT::RDF::RNode x_rdf,
                std::vector<std::string> cols, const size_t num_columns, 
                const size_t chunk_rows, const size_t start_row = 0, bool random_order=true) 
{
    
    // Fill the RTensor with the data from the RDataFrame
    ChunkLoader<float, float> func(x_tensor, num_columns, chunk_rows, random_order);
    auto myCount = x_rdf.Range(start_row, start_row + chunk_rows).Count();
    x_rdf.Range(start_row, start_row + chunk_rows).Foreach(func, cols);
    return myCount.GetValue();
}
""")