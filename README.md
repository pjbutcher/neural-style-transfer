# neural-style-transfer

![alt text](https://github.com/pjbutcher/neural-style-transfer/blob/master/example-outputs/pdx_1_starrynightrhone_clb1c1_sl15c2.png)

Merge the artistic style from one image to the content of another image using neural style transfer as described in the original paper by [Gatys 2015](https://arxiv.org/abs/1508.06576) 

## Motivation
Lesson 8 from part 2 of [fast.ai's online MOOC](http://course.fast.ai/) presented the code and technique to apply neural style transfer. I showed my wife what neural transfer was capable of and she immediately got excited and wanted to try out all sorts of styles on various images. Since the original code from the MOOC was presented in a Juypter notebook I decided to turn the code into a script which is callable from the command line. Now I can easily create all sorts of new images at my wife's request!

After some initial fun I got curious about tinkering around with some of the hard coded settings. For example, which VGG layers to use for the content/style recreation, how many interations to run, what weight to apply to each style layer, etc. So I ended up adding a few optional arguments which are described below.

## Requirements
* python 3+
* numpy
* keras
* tensorflow
* PIL
* scipy

Since this is a deep learning task a decent GPU is required. I did all testing on my local machine which has an Nvidia GeForce GTX 1070. 

## Basic Usage
python style_transfer.py content.png style.png --output_final stylized_img.png

## Example
The image at the beginning of this README was generated from the following content and style images (both are available in the content-imgs and style-imgs directories. I used the first blocks conv layer for the content recreation (fast.ai lesson 8 actually uses block 4's 2nd conv layer) since it gave much nicer results for these content/style images. 

python style_transfer.py content-imgs/pdx_1.jpg style-imgs/starrynightrhone.jpg --output_final results/pdx_1_starrynightrhone_clb1c1_sl15p --iters 10 --content_layer block1_conv1 

**Content - A nice photo of Portland's downtown waterfront (I'm from Portland):**
![alt text](https://github.com/pjbutcher/neural-style-transfer/blob/master/content-imgs/pdx_1.jpg)

**Style - Vincent van Gogh's Starry Night Over the Rhone:**
![alt text](https://github.com/pjbutcher/neural-style-transfer/blob/master/style-imgs/starrynightrhone.jpg)

**Output - Wow! It looks much nicer than I was expecting. It even attempts to put stars in the sky!:**
![alt text](https://github.com/pjbutcher/neural-style-transfer/blob/master/example-outputs/pdx_1_starrynightrhone_clb1c1_sl15c2.png)
