%%
%  Author:              Mohit Prabhushankar
%  PI:                  Ghassan AlRegib
%  Version:             1.0
%  Published in:        International Conference on Image Processing Sept 2019
%  Publication details: 

%%
% Demo script. img1 is the original image and img2 is the distorted image.
% Demo images are taken from TID2013 and read below

addpath 'Demo Images/'
addpath 'UNIQUE/'
addpath 'InputWeights/'

%%
img1 = imread('Demo Images/Original Image.BMP');
img2 = imread('Demo Images/Distorted Image.bmp');

%%
% Call mslUNIQUE which returns the perceived quality for activations.
% Calling mslGRADIENTS returns UNIQUE's corresponding gradient based perceived quality
% Note that both mslUNIQUE and mslGRADIENTS are exclusive and can be run independently.
% For UNIQUE please refer to : "https://arxiv.org/pdf/1810.06631.pdf"
% Value nearer to 1 represents a better quality image

activation_quality = mslUNIQUE(img1,img2)
gradient_quality = mslGRADIENTS(img1,img2)