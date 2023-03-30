# Generate binary ground-truth mask from control points using Hermite spline interpolation
This repository contains a simple Python script to generate binary segmentation masks from an arbitrary amount of control points, using Hermite spline interpolation. 



## Motivation
In image segmentation, it is often desirable to obtain binary ground-truth masks for training and evaluation of segmentation models. In some cases, such masks can be manually annotated, but this process can be time-consuming and expensive. This function provides a way to generate binary masks from control points, which are easier and cheaper to obtain.

## Functionality
The function takes as input a path to a text file that contains the coordinates of the control points (x values in the left column, y values in the right column). The annotation file can contain an arbitrary amount of control points, in any order. Additionally, the function also takes as input the desired width and height of the output binary mask.

The function first loads the control points from the file and ensures that the curve passes through the endpoints. It then computes tangents at each point and interpolates the curve using Hermite spline. The interpolated curve is then used to generate a binary mask of specified size, where all the pixel values inside the curve are 1 and 0 otherwise. Finally, the function uses a flood fill algorithm to fill the inside of the mask.

## Usage
The function only requires a few packages, namely `numpy`, `opencv` and `scipy`. Use the provided [environment.yaml](/environment.yaml) file for quick installation.

## Demonstration
Check the [provided notebook](/demo/demo.ipynb) for a quick demo on medical images. 
