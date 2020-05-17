# Text recognition 
Pytorch implementation for image-based sequence recognition tasks, such as scene text recognition and OCR.

# Paper
https://arxiv.org/pdf/1507.05717.pdf

# Demo
| demo images | VGG-BiLSTM-CTC | VGG-BiLSTM-CTC(case-sensitive) |
| ---         |     ---      |          --- |
| <img src="./demo_images/demo_1.png" width="300">     |   available   |    |
| <img src="./demo_images/demo_2.png" width="300">   |    professional   |       |
| <img src="./demo_images/demo_3.png" width="300">  |   londen   |     |
| <img src="./demo_images/demo_4.png" width="300">     |    greenstead    |     |
| <img src="./demo_images/demo_5.png" width="300">   |   weather   |     |
| <img src="./demo_images/demo_6.png" width="300">       |    please    |     |
| <img src="./demo_images/demo_7.png" width="300">   |   research   |   |
| <img src="./demo_images/demo_8.png" width="300" height="100"> |    grred    |      |

# Dataset
* Synth90k: 
  * **Introduction:** The Synth90k dataset contains 9 million synthetic text instance images from a set of 90k common English words. Words are rendered onto natural images with random transformations and effects, such as random fonts, colors, blur, and noises. Synth90k dataset can emulate the distribution of scene text images and can be used instead of real-world data to train data-hungry deep learning algorithms. Besides, every image is annotated with a ground-truth word.  
  * **Link:** [Synth90k-download](http://www.robots.ox.ac.uk/~vgg/data/text/)

# Dependence
* lmdb
* editdistance
* torch-summary
