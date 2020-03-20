# Capstone
## By Ludovica Schaerf

## Multi-label Hierarchical Classification of Paintings into their Represented Icon(s)


### Summary
This thesis will assess the applicability of a number of deep learning techniques to paintings classification, with the aim of automatically tagging paintings into the icon(s) they depict. A model will be trained and tested on the Tate Collection Dataset (Tategallery, tategallery/collection), a dataset of paintings containing 70000 images. In addition, a second model will be trained on the Brill Dataset (Posthumus, Brill), containing 87749 images, to evaluate the generalizability of the model to other iconographical classifications. The original model will use as target the ‘Subject’ metadata of the Tate Collection Dataset, a hierarchical multi-label structure encoding the icon(s) represented in each painting, at different levels of specificity. Due to the limited size of the dataset, the model will use weights pre-trained on ImageNet as activation to the network. 

The results of different multi-label, hierarchical and transfer learning techniques will be evaluated and a general reflection will be included at the end on how each of these methods performs at paintings classification. The performance will be discussed in view of the artistic contents of the paintings, keeping in mind that classification of artworks requires an increased level of abstraction in comparison to standard object recognition in images. The special case of abstract art will be inspected as a case study. 
This automatization of iconographical metadata production will hopefully contribute to alternative indexing of paintings, allowing to search paintings by content rather than only by artist, style and period. 
