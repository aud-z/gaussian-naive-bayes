This is a simple implementation of a Gaussian Naive Bayes model to classify brain image scans.
The model first optimizes performance by extracting the top k features, then uses MLE to estimate the model parameters.

To run the model from the command line, include the following arguments:
1) train_input: training input .csv file
2) test_input: test input .csv file
3) train_out: output file of predicted labels for training data
4) test_out: output file of predicted labels for test data
5) metrics_out: output file of evaluation metrics 
6) num_voxels: k for the top k features found via feature selection that will be used for prediction

