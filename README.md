# Capstone By Ludovica Schaerf

## Multi-label Hierarchical Classification of Paintings into their Represented Icon(s)

This repository contains the development work for my bachelor thesis.  In the thesis, I am classifying the artworks in the Tate Collection Dataset into the Subject Index (a thesaurus that indicates the contents in the image, similar to Iconclass). The classification is done at level 1 of the thesaurus (with 15 classes) and at level 2 (with 141 classes) and it is multi-label at both levels. 

The repository is organised as follows:
- **data**: contains all the manipulations of the data,
- **multi-label**: contains the classifications at level 1,
- **hierarchical**: contains the classifications at level 2. 

In **data** there are the following files:
- [Notebook on Dataset Description](data/FlatDatasetDescription.ipynb): contains some statistics on the dataset and on the target of the classification
- [Notebook on Pre-processing for the Hierarchical Datahandler](data/Pre-processingForHierarchicalDatahandler.ipynb): contains a sample of the preprocessing for the classification at level 2. If run, it pickles two csvs containing the preprocessed image paths and targets, which can be used later for classification without having to re-run the preprocessing 
- [Notebook on Pre-processing for the Multi-label Datahandler](data/Pre-processingForMulti-LabelDatahandler.ipynb): contains a sample of the preprocessing for the classification at level 1. If run, it pickles two csvs containing the preprocessed image paths and targets, which can be used later for classification without having to re-run the preprocessing 
- [Notebook on How to Retrieve the Dataset Images](data/RetrievingTateModernImages.ipynb): contains the code to retrieve the images from the urls that are saved on artwork_data.csv (among the files that can be downloaded from https://github.com/tategallery/collection
- [Notebook on Dataset Exploration](data/TateDatasetExploration.ipynb): contains an investigation into the files of the Tate collection and how they can be used to retrieve the information on the Subject Index.
- __init__.py
- [Python file containing Cascading Datahandler](data/datahandler_cascading.py): the datahandler that is used as a generator for all models at level 2. The handler resizes and batches the data
- [Python file containing Multi-label Datahandler](data/datahandler_multilabel.py): the datahandler that is used as a generator for all models at level 1. The handler resizes and batches the data

In **multi_label** there are the following files/folders:
- [Folder containing CNN-RNN implementation](multi-label/CNN_RNN/): a folder that contains the adapted implementation of CNN-RNN in pytorch, downloaded from https://github.com/Epiphqny/Multiple-instance-learning, the datahandler is fully written by me.
- [Notebook on Pre-processing for CNN-RNN](multi-label/CNN-RNN-flat-multi-label-classification.ipynb): contains all of the preliminary work that was done in order to be able to run the files in CNN-RNN.
- [Notebook on Flat Multi-label Classification](multi-label/FlatMulti-LabelClassification.ipynb): contains the classification code, with all the pretrained used, with and without finetuning and the scratch model.
- [Notebook containing Model Evaluations](multi-label/Model-Evaluation.ipynb): contains all the visualisations that have been made, together with most of the evaluations.
- [Python file for general evalution](multi-label/evaluate.py): this is a python file that loops over all the saved models and evaluates them according to accuracy, precision, recall, F1 score both overall and per-class.
- [Python file containing flat multi-label classification](multi-label/flat_multi_label_classification.py): a python version of FlatMulti-LabelClassification.ipynb, the file automatised and takes the model, whether to fine tune or not and whether to add weights as argument.
- [Python file containing scratch model architecture](multi-label/out_of_the_box.py): contains the architecture of the baseline model, which is trained from scratch.

In **hierarchical** there are the following files:
- [Notebook on Cascading Classification](hierarchical/Cascading-Classification.ipynb): implements a cascading classification that uses the prediction at level 1 and embeds it into an input for the prediction at level 2.
- [Notebook on Hierarchical Classification](hierarchical/Hierarchical-Classification.ipynb): implements HDCNN according to https://github.com/justinessert/hierarchical-deep-cnn.
- [Python file for flat multi-label classification at level 2](hierarchical/deep_multi_label_classification.py): file equivalent to flat_multi_label_classification.py but with the data of level 2. Here the classification of level 2 does not use any information of the prediction of level 1.
