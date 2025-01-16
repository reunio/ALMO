# Active Learning-based Molecular Optimization(ALMO)

The complete code will be released after receive, but you can try the trained model for QED optimization.

You can find model file from: https://drive.google.com/file/d/1n7YSigJCs7jmPFuP9eXaLS-l7UcLd-Vh/view?usp=sharing

## Requirements
* python=3.8.13 
* scikit-learn==0.23.1
* accelerate==0.34.2
* transformers==4.31
* charset-normalizer==3.3.2
* rdkit==2023.3.2
* libsvm-official==3.32.0
* pandas=1.4.4


## Test
To test a trained ALMO, run

    python 4ES_test.py --epochs_test 50 --swarm_num 30 --min_similar 0.6 --data_fname_test data/standard_test_data/qed_test.csv --test_model_path KLmodel_QED

where the parameters are:
* epochs_test: the number of epochs to test.
* swarm_num: the number of swarm.
* min_similar: the minimum similarity.
* data_fname_test: the path of test data.
* test_model_path: the path of trained model.



