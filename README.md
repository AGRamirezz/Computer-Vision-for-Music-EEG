# Computer-Vision-for-Music-EEG
The goal for this project is to develop methods that leverage standard computer vision architectures for classification of long brain responses to music stimuli. 

This research is a follow up to EEG-Music classification work done by (Sonawane et al., 2021) where they cleverly use spectral component extraction to format EEG as 2D input for CNN architectures to correctly identify the song name that a participant was listening to from a 1 second EEG example. 

Our project unwraps the above mentioned findings and tests how different input representations of EEG can be thought of as types of images in traditional computer vision methods. We show SOTA performance without the need of a feature extraction step, as a first example of end-to-end models using EEG responses to long music stimuli. Transfer learning with Resnet also shows a first in SOTA performance with EEG data transferred to models pretrained under a different domain.  

Currently under review at ICASSP 2022. Code and results public release are pending. 
