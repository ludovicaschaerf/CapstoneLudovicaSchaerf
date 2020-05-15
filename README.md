# Capstone By Ludovica Schaerf

## Multi-label Hierarchical Classification of Paintings into their Represented Icon(s)

This repository contains the development work for my bachelor thesis.  In the thesis, I am classifying the artworks in the Tate Collection Dataset into the Subject Index (a thesaurus that indicates the contents in the image, similar to Iconclass). The classification is done at level 1 of the thesaurus (with 15 classes) and at level 2 (with 141 classes) and it is multi-label at both levels. 

The repository is organised as follows:
- **data**: contains all the manipulations of the data,
- **multi-label**: contains the classifications at level 1,
- **hierarchical**: contains the classifications at level 2. 

In **data** there are the following files:
- FlatDatasetDescription.ipynb: this contains some statistics on the dataset and on the target of the classification
- Pre-processingForHierarchicalDatahandler.ipynb: this contains a sample of the preprocessing for the classification at level 2. If run, it pickles two csvs containing the preprocessed image paths and targets, which can be used later for classification without having to re-run the preprocessing 
- Pre-processingForMulti-LabelDatahandler.ipynb: this contains a sample of the preprocessing for the classification at level 1. If run, it pickles two csvs containing the preprocessed image paths and targets, which can be used later for classification without having to re-run the preprocessing 
- RetrievingTateModernImages.ipynb: this contains the code to retrieve the images from the urls that are saved on artwork_data.csv (among the files that can be downloaded from https://github.com/tategallery/collection
- TateDatasetExploration.ipynb: this contains an investigation into the files of the Tate collection and how they can be used to retrieve the information on the Subject Index.
- __init__.py
- datahandler_cascading.py: this is the datahandler that is used as a generator for all models at level 2. The handler resizes and batches the data
- datahandler_multilabel.py: this is the datahandler that is used as a generator for all models at level 1. The handler resizes and batches the data

In **multi_label** there are the following files/folders:
- CNN_RNN: this is a folder that contains the adapted implementation of CNN-RNN in pytorch, downloaded from https://github.com/Epiphqny/Multiple-instance-learning, the datahandler is fully written by me.
- CNN-RNN-flat-multi-label-classification.ipynb: this file contains all of the preliminary work that was done in order to be able to run the files in CNN-RNN.
- FlatMulti-LabelClassification.ipynb: this contains the classification code, with all the pretrained used, with and without finetuning and the scratch model.
- Model-Evaluation.ipynb: this file contains all the visualisations that have been made, together with most of the evaluations.
- evaluate.py: this is a python file that loops over all the saved models and evaluates them according to accuracy, precision, recall, F1 score both overall and per-class.
- flat_multi_label_classification.py: this is a python version of FlatMulti-LabelClassification.ipynb, the file automatised and takes the model, whether to fine tune or not and whether to add weights as argument.
- out_of_the_box.py: this contains the architecture of the baseline model, which is trained from scratch.

In **hierarchical** there are the following files:
- Cascading-Classification.ipynb: this implements a hierarchical classification that uses the prediction at level 1 and embeds it into an input for the prediction at level 2.
deep_multi_label_classification.py: this file is equivalent to flat_multi_label_classification.py but with the data of level 2. Here the classification of level 2 does not use any information of the prediction of level 1.
