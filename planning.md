
1. Redesign code
    - Make the code flexible in number of columns and rows 
    - Reduce redundency 
    - Randomize the data instead always the same batches

2. Pythonize the RTensor
3. Write tutorials and demos
4. Parralelize the reading and ML 

5. (extension) Use data that is not flat, but more complex
6. (extension) Optimize for GPU machine learning

Step 1 - 4 done end of january

step 5 and 6 will be worked on in february, March

## Project 2

I/O parameters optimize for throughput. 
    - most are categorical
    - some binary
    - some continuous

    - different groups of parameters
        - Compression
        - Decompression

Side constrains maybe be added as well (memory and filesize). 

Some parameters are connected on eachother. 

Explainability is important 

similar to Hyperparameter optimization

1. Building data pipeline (3 months)
    - Already benchmarks available
    - (Goal) Need to figure out how to input specific parameter configurations
    - (Goal) Define format for results
    - (Goal) Create a set of different paramter configurations

2. Analysis (1 month) (many extensions possible)
    - Decision Trees
    - Complex algorithms
    - Finding universally good parameters


## Benchmarking

1. Loading random vs loading sequential
    - Different batch sizes
    - Different chunk sizes 
2. Batching random vs sequential
    - Different batch sizes
    - Different chunk sizes 
3. NDataFrame vs DatasetSpec
4. Python vs C++