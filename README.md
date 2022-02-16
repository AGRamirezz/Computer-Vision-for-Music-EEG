# Computer-Vision-for-Music-EEG
In this study we show that EEG responses of participants listening to music can be efficiently processed by standard computer vision models. Our results produce SOTA performance on an end-to-end classification task.

Paper: https://arxiv.org/abs/2202.03265?context=eess

Cite As: Ramirez-Aristizabal, A. G., Ebrahimpour, M. K., & Kello, C. T. (2022). Image-based eeg classification of brain responses to song recordings. arXiv preprint arXiv:2202.03265.

Content Summary:

9 Models 

2 Datasets

-Natural Music EEG Dataset - Tempo

--Link: https://exhibits.stanford.edu/data/catalog/jn859kj8079

--Cite As: Losorelli, S., Nguyen, D. T., Dmochowski, J. P., & Kaneshiro, B. (2017, October). NMED-T: A Tempo-Focused Dataset of Cortical and Behavioral Responses to Naturalistic Music. In ISMIR (pp. 339-346).

-Natural Music EEG Dataset - Hindi

--Link:https://exhibits.stanford.edu/data/catalog/sd922db3535

--Cite As: Kaneshiro, B., Nguyen, D. T., Dmochowski, J. P., Norcia, A. M., & Berger, J. (2016). Naturalistic music EEG dataset—hindi (nmed-h). In Stanford Digital Repository.. Stanford Digital Repository.


This research is a follow up to EEG-Music classification work done by (Sonawane et al., 2021) where they cleverly use spectral component extraction to format EEG as 2D input for CNN architectures to correctly identify the song name that a participant was listening to from a 1 second EEG example. 

Our project unwraps the above mentioned findings and tests how different input representations of EEG can be thought of as types of images in traditional computer vision methods. We show SOTA performance without the need of a feature extraction step, as a first example of end-to-end models using EEG responses to long music stimuli. Transfer learning with Resnet also shows a first in SOTA performance with EEG data transferred to models pretrained under a different domain.  

 
