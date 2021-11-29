# State Farm Distracted Driver Detection - Can computer vision spot distracted drivers?

## Overview

According to the CDC motor vehicle safety division, one in five car accidents is caused by a distracted driver. Sadly, this translates to 425,000 people injured and 3,000 people killed by distracted driving every year.

State Farm hopes to improve these alarming statistics, and better insure their customers, by testing whether dashboard cameras can automatically detect drivers engaging in distracted behaviors. Given a dataset of 2D dashboard camera images, State Farm is challenging Kagglers to classify each driver's behavior. Are they driving attentively, wearing their seatbelt, or taking a selfie with their friends in the backseat?

Here is the link for it(https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview/description).

## Applied methods

+ Data Augmentation

We applied basic data warping augmentation methods, color jittering and random resized cropping. Since there is geometrically symmetric different labels, we exclude gemotric transformation such as flipping, rotating. 

+ Number of layers of ResNet

We find the effect of residual skip connection of ResNet to solve the vanishing gradient problem[1]. 

+ Applying SE

SEResNet is the model that applied Squeeze-and-Excitation(SE) to ResNet[2]. SE improves performance through readjustment considering the importance per channel characteristics generated through convolution in the convolution network.

+ Dropout

We applied the dropout tech to the Fully-Connected(FC) layers to prevent overfitting.

+ Optimizer

We examined various optimizer. Popular optimizers such as Stochastic Gradient Descent(SGD) and Adam are tested. Also, state of art optimizer for image classification, Sharpneess Aware Minimization(SAM) is also tested[3]. It makes it more efficient to reach the global minimum by adding a process that makes the local minimum smooth.

+ Learning Rate Scheduler

We used popular schedulers, 'stepLR' and 'cosineAnnealingLR'. Also, 'CosineAnnealingWarmUpRestarts' in the spotlight was used[4]. This is an improved version of the Cosine scheduler, where the period of amplitude gradually decreases, and the period also gradually increases as the epoch increases.

## Final Model

Final model configurations are below.

> Network Model : ResNet 50
> Optimizer : SAM
> Augmentation : ColorJitter(), RandomResizedCrop(), Normalization()
> LR Scheduler : CosineAnnealingWarmUpRestarts()
> Dropout : p=0.5
> Learning Rate : 1e-4
> Batch Size : 16
> Weight Decay : 5e-4


## Results

Submissions are evaluated using the multi-class logarithmic loss. Our score is 0.41048, which is the top 20% score.

## Further Apporach

+ Various data augmentation such as Autoaugment[5]

+ Dropout for the hidden layers

+ Transfer learning

+ Accuracy based schedulers

## References

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[2]  HU, Jie; SHEN, Li; SUN, Gang. “Squeeze-and-excitation networks”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018. p. 7132-7141.

[3] Kwon, Jungmin, et al. "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks." arXiv preprint arXiv:2102.11600 (2021).

[4] Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. arXiv: Learning.

[5] Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., & Le, Q. V. (2018). Autoaugment: Learning augmentation policies from data. arXiv preprint arXiv:1805.09501.