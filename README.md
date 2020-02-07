# Capstone
## By Ludovica Schaerf

Semantic Classification of Paintings According to their Represented Subject(s)
 
Is a subject classifier of artworks useful? How so?
What is an approriate structure of classification to categorize following a hierarchical label notation?
How well can an algorithm detect the content of an artwork? How much does adding a classifier pre-trained on real world images help? Til what level of specificity is the prediction accurate?
What do the results tell us about the ability of machines to make judgements on works of art? How does it perform compared to humans?
 
 
Motivation:
 
Datasets:
OmniArt using Iconclass
(Strezoski, Gjorgji, and Marcel Worring. ‘OmniArt: A Large-Scale Artistic Benchmark’. ACM Transactions on Multimedia Computing, Communications, and Applications 14, no. 4 (23 October 2018): 1–21. https://doi.org/10.1145/3273022.)
Tate Gallery using subject notation (https://github.com/tategallery/collection, https://www.tate.org.uk/art/archive) 
Methods:
Hierarchical Deep Learning
Multi-label classification
Parallelized Deep Learing
Learning given an image and a caption
Adding pre-trained models
Example Format Models:
https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
https://github.com/satyatumati/Hierarchical-Deep-CNN/blob/master/code/model.ipynb
https://github.com/justinessert/hierarchical-deep-cnn/blob/master/hdcnn.ipynb
https://keras.io/applications/
https://www.tensorflow.org/tutorials/text/image_captioning
https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
https://nlp.stanford.edu/projects/glove/
Research Context:
Lincoln, Matthew; Levin, Golan; Conell, Sarah Reiff; Huang, Lingdong (2019): National Gallery of Art InceptionV3 Features. figshare. Dataset. https://doi.org/10.1184/R1/10061885.v2
Elgammal, Ahmed, Marian Mazzone, Bingchen Liu, Diana Kim, and Mohamed Elhoseiny. ‘The Shape of Art History in the Eyes of the Machine’. ArXiv:1801.07729 [Cs], 12 February 2018. http://arxiv.org/abs/1801.07729.
Noord, Nanne van, Ella Hendriks, and Eric Postma. ‘Toward Discovery of the Artist’s Style: Learning to Recognize Artists by Their Artworks’. IEEE Signal Processing Magazine 32, no. 4 (July 2015): 46–54. https://doi.org/10.1109/MSP.2015.2406955.
Saleh, Babak, Kanako Abe, Ravneet Singh Arora, and Ahmed Elgammal. ‘Toward Automated Discovery of Artistic Influence’. Multimedia Tools and Applications 75, no. 7 (April 2016): 3565–91. https://doi.org/10.1007/s11042-014-2193-x.
Sigaki, Higor Y. D., Matjaž Perc, and Haroldo V. Ribeiro. ‘History of Art Paintings through the Lens of Entropy and Complexity’. Proceedings of the National Academy of Sciences 115, no. 37 (11 September 2018): E8585–94. https://doi.org/10.1073/pnas.1800083115.
Taylor, Richard P., Adam P. Micolich, and David Jonas. ‘Fractal Analysis of Pollock’s Drip Paintings’. Nature 399, no. 6735 (June 1999): 422–422. https://doi.org/10.1038/20833.
Manovich, Lev. ‘Data Science and Digital Art History’. Digital Art History, no. 1 (2015).
———. ‘Defining AI Arts: Three Proposals’, 2019, 9.
Visual Link Retrieval in a Database of Paintings Benoit Seguin(B) , Carlotta Striolo, Isabella diLenardo, and Frederic Kaplan
Large-Scale Object Classification using Label
Relation Graphs Jia Deng †∗ , Nan Ding ∗ , Yangqing Jia ∗ , Andrea Frome ∗ , Kevin Murphy ∗ , Samy Bengio ∗ , Yuan Li ∗ , Hartmut Neven ∗ , Hartwig Adam ∗
HD-CNN: Hierarchical Deep Convolutional Neural Network for Large Scale
Visual Recognition Zhicheng Yan †
Descriptive Metadata, Iconclass, and Digitized Emblem Literature Timothy W. Cole
Computational Modeling of Affective Qualities of Abstract Paintings
H I N ET : H IERARCHICAL C LASSIFICATION WITH NEURAL N ETWORK Sean Saito, Zhenzhou Wu
Computational aesthetics and applications Yihang Bo
Computational Art A. Eliëns.
Panofsky and ICONCLASS Roelof Van Straten
Hierarchical Multi-Label Classification Networks Jônatas Wehrmann
Long-term Recurrent Convolutional Networks for Visual Recognition and Description Jeff Donahue
Recognizing Art Style Automatically with deep learning Florian Yger
Aesthetics of Neural Network Art Aaron Hertzmann https://arxiv.org/abs/1903.05696 
 
Tate Dataset
The dataset adopted by this paper is the publicly-available Tate Collection (https://github.com/tategallery/collection), containing the metadata and image urls of over 70,000 artworks owned by the Tate or jointly with the National Galleries of Scotland as part of ARTIST ROOMS. The repository contains both basic metadata in csv format and some more elaborate metadata (including the subjects represented in each artwork) in json files. Of the 70,000 artworks in the metadata, over 40,000 corresponding images were easily retrievable online from the url provided. 

The collection has been published on github as part of the 5-year project of digital access, participation and learning with archives that started in 2012. The metadata has been dynamically updated until October 2014, when the repository stopped being actively maintained. The project, Achives & Access, has been funded by the Heritage Lottery Fund (HLF) and Tate. It focused on the digitalisation of the Tate Collection (which is currently the world’s largest archive of British art) and the development of interactive actvities at Tate. 

(Make table like this)



The metadata concerning the subject(s) of the artworks is used by this paper as the target for the classification. An example of this target is the following:


This example is the subject metadata (which can be found under the section ‘EXPLORE’ at the bottom of each image of the Tate digital archive) belongs to:

 
from: https://www.tate.org.uk/art/artworks/warhol-mick-jagger-ar00428

As one can see from this example, the subject metadata is extremely detailed. It is structured in the form of a tree in which each applicable node has a variable number of children and the children can also have children. Empirically, I observed that the biggest depth is indeed 3 (where the root has depth 0). The nodes at depth one denote radically different aspects, both present in the image and not. For instance, ‘people’ refers to the content of the image (indeed a portrait of Mick Jagger), while ‘places’, in this case, are not visible in the image itself and are therefore retrieved from other sources. This extremely varied nature of the subject metadata makes a classification based merely on the image as input extremely difficult. Some pre-processing and filterning is therefore necessary: the idea is to delete the nodes that refer to data that cannot be found in the image itself. Unfortunately, this is rather complicated as the same nodes may at times refer to something in the image and at times not. 

In terms of the classification, each node is in a 1-to-many relation to its children, which indicates that the classification at each level needs to be multi-label or that only of the children has to be selected (in which case, how do I select the most significant one?). Furthermore, as already discussed, the subject metadata has depth 3, which implies that the classification can either be hierarchical / cascading or that the classification has to be done at the level of the leaves, making the nature of the classification even more multi-labelled. 

The subject metadata is contained in the dataset in the form of a subsection of the json file that corresponds to each artwork. The following is an example of the subsection:
   
As one can see, in addition to what was already in the website, each name has a corresponding numeric id. There is no clear pattern that indicates how this code has been made, but more work has to be put in the investigation of this taxonomy. Further, the code corresponding to the section is used by the url leading to the pages of each image. 


