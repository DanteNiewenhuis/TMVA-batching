## Deliverables

- Dataloader -> A functor that loads the content of a RDataFrame into a RTensor. 
                Using the Range function a specific chunk of the RDataFrame can be loaded.  
                The process is much faster than the current TMVA AsTensor implementation (around 15x). 
                


- BatchGeneratorHelper -> Takes a RTensor and splits it into batches. 
                          The generator can return both random batches, or sequential. 
                          However, returning random batches is standard in AI, and thus better to use. 
                          Comparing random batching to sequential batching has not resulted in significant time difference yet (more testing needed).

- BatchGenerator -> Combines the Dataloader and the BatchGeneratorHelper. 
                        A Chunk of Data is loaded into memory using the DataLoader. 
                        After, the BatchGeneratorHelper is used to create batches. 
                        When the current chunk of data has been loaded completely, a new chunk is loaded. 

                    The user can specify the size of a chunk. This is especially useful when dealing with larger files. 

## Problems

- Column number ->  Problem: The implementation of the DataLoader uses templating to read the columns. 
                    Solution: Dynamicly compile the DataLoader functions.
                        This already works in the Python implementation, but still has to be added to the Cpp loader

- DataSetSpec -> Currently, loading chunks gets slower further into the process. This is because the Range function first has through all the rows 
                to get to the starting row. A way to fix this would be to use DatasetSpec instead of RDataFrame directly. 
                This seems to be much faster, but has the problem that it needs to reload the RDataFrame for every chunk. 
                It is also more difficult to combine it with filters. I will have to do some tests to know which method to choose. 

- Parralel -> Ideally, a new chunk can be loaded into memory while processing the batches. 
              An initial version of this has already been implemented in Python, but can be improved.


- DataSetSpec cannot load rows that are not in the RDataFrame. We will thus need another method of stopping