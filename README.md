# Survey on Quantization in ML

Literature survey of Quantization in the context of ML for reducing model size, inference time, and computations.

## Page Structure
- Important Conferences [TODO]
- Important People and Groups [TODO]
- Important Terminology [TODO]
- Important Papers
- Uses [shield.io](https://shields.io/) badges to show information at a glance (ex: ![](https://img.shields.io/badge/impact--factor-33.49-purple), ![](https://img.shields.io/badge/type-hardware-orange.svg))

## Important Conferences

```
todo
```
<!-- 
### NeurIPS  ![](https://img.shields.io/badge/impact--factor-33.49-purple)
- Neural Information Processing Systems
- Dec 6, 2021 - Dec 14, 2021 - Online
- Premier machine learning conference
 -->

## Important People and Groups:

#### Song Han [![](https://img.shields.io/badge/h--index-40-blue.svg)](https://scholar.google.com/citations?user=E0iCaa4AAAAJ&hl=en&oi=ao)

```
todo: add more information about authors (key contributions, etc.)
```
## Important Terminology

#### Precision
##### Floating Point
- Varying number of digits after decimal point
##### Fixed Point
- Fixed number of digits after the decimal point
- Arithmetic is done with integers

#### Huffman Coding

```
todo
```
<!-- ### Attack types based on DNN model knowledge
#### White-box attacks
#### Black-box attacks

### Types based on attack setting
#### Theoretical
#### Digital (usually against image classifiers)
#### Simulation
#### Physical -->

## Important Papers

#### Han, Song, Huizi Mao, and William J. Dally. "Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding." arXiv preprint arXiv:1510.00149 (2015). [link](https://arxiv.org/pdf/1510.00149.pdf) 
![](https://img.shields.io/badge/dataset-ImageNet-orange.svg) ![](https://img.shields.io/badge/models-AlexNet_|_VGG--16-green.svg) 
- Three step process:
  - Pruning - learn only important network connections
  - Quantization - quantized centroids for weight sharing
  - Huffman coding

This paper gives a strong motivation for the need for energy savings on mobile devices with numerical backing:

> Under 45nm CMOS technology, a 32 bit
floating point add consumes 0.9pJ, a 32bit SRAM cache access takes 5pJ, while a 32bit DRAM memory access takes 640pJ, which is 3 orders of magnitude of an add operation. Large networks do not fit in on-chip storage and hence require the more costly DRAM accesses. Running a 1 billion connection neural network, for example, at 20fps would require (20Hz)(1G)(640pJ) = 12.8W just for DRAM access - well beyond the power envelope of a typical mobile device.

#### Iandola, Forrest N., Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size." arXiv preprint arXiv:1602.07360 (2016). [link](https://arxiv.org/pdf/1602.07360.pdf)

#### Han, Song, Jeff Pool, John Tran, and William J. Dally. "Learning both weights and connections for efficient neural networks." arXiv preprint arXiv:1506.02626 (2015). [link](https://arxiv.org/pdf/1506.02626.pdf)

#### Zhang, Tianyun, Shaokai Ye, Kaiqi Zhang, Jian Tang, Wujie Wen, Makan Fardad, and Yanzhi Wang. "A systematic dnn weight pruning framework using alternating direction method of multipliers." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 184-199. 2018. [link](https://arxiv.org/pdf/1804.03294v3.pdf)

#### Anwar, S., Hwang, K. and Sung, W., 2015, April. Fixed point optimization of deep convolutional neural networks for object recognition. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1131-1135). IEEE. 

#### Lin, Darryl, Sachin Talathi, and Sreekanth Annapureddy. "Fixed point quantization of deep convolutional networks." In International conference on machine learning, pp. 2849-2858. PMLR, 2016. [link](http://proceedings.mlr.press/v48/linb16.pdf) ![](https://img.shields.io/badge/dataset-CIFAR--10_|_ImageNet-orange.svg)
- Proposes algorithm to convert floating point trained Deep Convolutional Network (DCN) to fixed point 
- Converts CNN layer activations and weights into fixed point
- Builds upon Sajid et al (2015)'s work on converting pretrained floating point networks to fixed point model using an optimization stratgy based on signal-to-quantization-noise-ratio (SQNR) instead of exhaustive search.
- Range = Stepsize * 2 ^ Bitwidth
  - have a large enough range to reduce overflow
  - small enough resolution to reduce quantization error
> For example, suppose the input is Gaussian distributed with zero mean and unit variance. If we need a uniform quantizer with bit-width of 1 (i.e. 2 levels), the best approach is to place the quantized values at -0.798 and 0.798. In other words, the step size is 1.596. If we need a quantizer with bit-width of 2 (i.e. 4 levels), the best approach is to place the quantized values at -1.494, -0.498, 0.498, and 1.494. In other words, the step size is 0.996 ... sometimes it is desirable to have 0 as one of the quantized values because of the potential savings in model storage and computational complexity. This means that for a quantizer with 4 levels, the quantized values could be -0.996, 0.0, 0.996, and 1.992.
- gamma_dB = k * beta
  - gamma_db = 10*log10(gamma) -> SQNR in dB
  - k -> quantization efficiency
  - beta -> bit-width
  - k {uniform distribution} = 6dB/bit (practically 2-4dB/bit)
  - k {Gaussian distribution} = 5dB/bit
- Any floating point DCN model can be converted to fixedpoint by following these steps:
  - Run a forward pass in floating point using a large set of typical inputs and record the activations.
  - Collect the statistics (zeta) of weights, biases and activations for each layer.
  - Determine the fixed point formats (resolution) of the weights, biases and activations for each layer
    - step_size = zeta * Stepsize(beta) from the Table
  - Compute number of fractional bits n = round(-log2(step_size))
> The effect of quantization (degradation) can be accurately captured in a single quantity, the SQNR.
> the SQNR at the output of a layer in DCN is the Harmonic Mean of the SQNRs of all preceding quantization steps. Since the output SQNR is the harmonic mean, the network performance will be dominated by the worst quantization step.
> doubling the depth of a DCN will half the output SQNR (3dB loss). But this loss can be readily recovered by adding 1 bit to the bit-width of all weights and activations, assuming the quantization efficiency is more than 3dB/bit. (emprically unverified)
- Effects of DCN components
  - Batch Norm -> linear transformation that can be absorbed by neighboring CONV layer -> does not need to be modeled for quantization
  - ReLU -> ReLU only starts to affect SQNR calculation when the perturbation caused by quantization is sufficiently large to alter the sign of an activation
  - Non-ReLU activations -> Not modelled
- Not all layers have to be quantized with the same bit-width; layers that are more data dense would benefit from higher resolution than others
- Mininum bit-width before performance degradation is 6
> it is worth nothing that the proposed cross-layer layer bit-width optimization algorithm is most effective when the network size is dominated by convolutional layers, and is less effective otherwise.

Key takeaways:
- Be able to calculate the SQNR when performing quantization
- Being able to set contraints for particular research problems (in this case, the paper focuses on quantizing just the CONV layers, keeps FC layers for others to tackle)


#### Zhu, Chenzhuo, Song Han, Huizi Mao, and William J. Dally. "Trained ternary quantization." arXiv preprint arXiv:1612.01064 (2016). [link](https://arxiv.org/pdf/1612.01064.pdf)

#### Courbariaux, Matthieu, Itay Hubara, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. "Binarized neural networks: Training deep neural networks with weights and activations constrained to+ 1 or-1." arXiv preprint arXiv:1602.02830 (2016).
![](https://img.shields.io/badge/dataset-MINST_|_CIFAR--10_|_SVHN-orange.svg) 
- Contributions
  - Binarized Neural Networks (BNNs) -> neural networks with binarized weights and activations at run-time
  - Implemented on Torch7 and Theano on MNIST, CIFAR-10 and SVHN
  - Forward pass shows significant reduction in memory consumption and most arithmetic operations can be replaced with bit-wsie operations.
  - Programed a binary matrix multiplication GPU kernel that has 7x speedup than unoptimized GPU kernel
> When training a BNN, we constrain both the weights and the activations to either +1 or −1.
- Binarization functions
  - Deterministic: x<sup>b</sup> = sign(x)
  - Stochastic: x<sup>b</sup> = +1 with probability p = sigma(x) and -1 with probability 1 - p
    - where sigma is the hard sigmoid
- Since sign(x) is a non-differentiable function, they use something called a [straight-through estimator](https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html) for backpropagation. Essentially the gradients of the previous layer are backpropagated directly (via a hard tanh to clamp gradients to -1 to 1).
- Revamped DNN components
  - Shift based Batch Normalization
  - Shift based AdaMax for ADAM
  - First (image) layer is converted to 8-bit fixed point (needs verification)
> Although BNNs are slower to train, they are nearly as accurate as 32-bit float DNNs.
> filter replication is very common ... on our CIFAR-10 ConvNet, only 42% of the filters are unique.
> In BNNs, both the activations and the weights are constrained to either −1 or +1. As a result, most of the 32-bit floating point multiply-accumulations are replaced by 1-bit XNOR-count operations.

Key takeaways:
- Use the idea of straight-through estimator when dealing with non-differentiable functions
- Use of [hardtanh](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html) as an 'activation' to clamp values between -1 and 1 

#### Rastegari, Mohammad, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. "Xnor-net: Imagenet classification using binary convolutional neural networks." In European conference on computer vision, pp. 525-542. Springer, Cham, 2016.  [link](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_32)
![](https://img.shields.io/badge/dataset-ImageNet-orange.svg) ![](https://img.shields.io/badge/models-AlexNet-green.svg)
- 84% top1 accuracy on ImageNet with AlexNet, and out performs BinaryNet (Courbariaux et. al.)
- Two approximations
  - Binary-weight-networks (weight values are binary and convolution can be done with addition and subtraction without multiplication)
  - XNOR-networks (both weights and inputs to the CONV and FC layers are binary) -> FC are implemented as CONV


