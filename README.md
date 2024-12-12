# NN Dataset Results

## Breast Cancer Dataset

The folder titled **"NN_Dataset_1_Cancer"** contains all the neural network-based results for the Breast Cancer Dataset. Inside this folder:

- `breast-cancer.data`
- `breast-cancer.names`
- `Index`

are the downloaded raw data files. After running the `prepare_dataset.py` script, the preprocessed dataset files—`X_train.npy`, `X_test.npy`, `y_train.npy`, and `y_test.npy`—will be created in the same directory. Finally, after running the `NN.py` script, the final results will be printed, and the figure titled `nn_architecture_metrics_dataset_1.png`, summarizing the results, will be saved in the same directory and displayed.

## Indian Diabetes Dataset

The folder titled **"NN_Dataset_2_Diabetes"** contains all the neural network-based results for the Indian Diabetes Dataset. Inside this folder:

- `pima-indians-diabetes.csv`

is the downloaded raw data file. After running the `prepare_dataset.py` script, the preprocessed dataset files—`X_train.npy`, `X_test.npy`, `y_train.npy`, and `y_test.npy`—will be created in the same directory. Finally, after running the `NN.py` script, the final results will be printed, and the figure titled `nn_architecture_metrics_dataset_2.png`, summarizing the results, will be saved in the same directory and displayed.
