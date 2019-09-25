# Distorted Representation Space Characterization Through Backpropagated Gradients

[Gukyeong Kwon](https://www.linkedin.com/in/gukyeong-kwon/), [Mohit Prabhushankar](https://www.linkedin.com/in/mohitps/), [Dogancan Temel](http://cantemel.com/)and [Ghassan AlRegib](http://www.ghassanalregib.info)

This repository includes the codes for the paper:

Gukyeong Kwon*, Mohit Prabhushankar*, Dogancan Temel, and Ghassan AlRegib, "Distorted Representation Space Characterization Through Backpropagated Gradients", 2019 IEEE International Conference on Image Processing (ICIP), Taipei, Taiwan, 2019, pp. 2651-2655. (*: equal contribution) [[arXiv]](https://arxiv.org/abs/1908.09998) [[IEEE]](https://ieeexplore.ieee.org/abstract/document/8803228)

This paper received ***Best Paper Award*** in ICIP 2019.

--------
## Abstract
In this paper, we utilize weight gradients from backpropagation to characterize the representation space learned by deep learning algorithms. We demonstrate the utility of such gradients in applications including perceptual image quality assessment and out-of-distribution classification. The applications are chosen to validate the effectiveness of gradients as features when the test image distribution is distorted from the train image distribution. In both applications, the proposed gradient based features outperform activation features. In image quality assessment, the proposed approach is compared with other state of the art approaches and is generally the top performing method on TID 2013 and MULTI-LIVE databases in terms of accuracy, consistency, linearity, and monotonic behavior. Finally, we analyze the effect of regularization on gradients using CURE-TSR dataset for out-of-distribution classification. 


## Citation: 

If you have found our code and data useful, we kindly ask you to cite our work. You can cite the arXiv preprint for now: 
```tex
@inproceedings{kwon2019distorted,
  title={Distorted Representation Space Characterization Through Backpropagated Gradients},
  author={Kwon, Gukyeong and Prabhushankar, Mohit and Temel, Dogancan and AlRegib, Ghassan},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
  pages={2651--2655},
  year={2019},
  organization={IEEE}
}
```
## Questions?

The code and data are provided as is with no guarantees. If you have any questions, regarding the dataset or the code, you can contact the authors (gukyeong.kwon@gatech.edu or mohit.p@gatech.edu), or even better open an issue in this repo and we'll do our best to help.

