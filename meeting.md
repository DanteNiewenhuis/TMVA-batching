# PR

## TODO:

1.  Add tests
2.  Add tutorials
3.  Restructure code to be in line with other ROOT code
4.  Add stopping mechanism for validation batches (memory problem!!!)
5.  Improve handling of batching when there are not enough events left in a chunk to make a batch
    For now: Create smaller batch for the final batch of a chunk
6.  Implement sequential vs random batches 
7.  Look at the crash when generator is never used and Python exits


# Extensions and Improvements

1. Loading directly to GPU
2. Look into adding events from next chunk if there are too few events for a batch
3. Look into explicit template instantiation
4. Stop reloading of chunks if chunk is the size of the dataset
5. Start loading first chunk during validation
6. Look into separate non-filtered loading
7. Add the option to difine new features
8. Add support for loading multiple datasets