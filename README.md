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

#### Han, Song, Huizi Mao, and William J. Dally. "Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding." arXiv preprint arXiv:1510.00149 (2015). [link](https://arxiv.org/pdf/1510.00149.pdf) ![](https://img.shields.io/badge/dataset-ImageNet-orange.svg) ![](https://img.shields.io/badge/models-AlexNet_|_VGG--16-green.svg) 
- Three step process:
-- Pruning - learn only important network connections
-- Quantization - quantized centroids for weight sharing
-- Huffman coding

#### Iandola, Forrest N., Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size." arXiv preprint arXiv:1602.07360 (2016). [link](https://arxiv.org/pdf/1602.07360.pdf)

#### Han, Song, Jeff Pool, John Tran, and William J. Dally. "Learning both weights and connections for efficient neural networks." arXiv preprint arXiv:1506.02626 (2015). [link](https://arxiv.org/pdf/1506.02626.pdf)

#### Zhang, Tianyun, Shaokai Ye, Kaiqi Zhang, Jian Tang, Wujie Wen, Makan Fardad, and Yanzhi Wang. "A systematic dnn weight pruning framework using alternating direction method of multipliers." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 184-199. 2018. [link](https://arxiv.org/pdf/1804.03294v3.pdf)

#### Lin, Darryl, Sachin Talathi, and Sreekanth Annapureddy. "Fixed point quantization of deep convolutional networks." In International conference on machine learning, pp. 2849-2858. PMLR, 2016. [link](http://proceedings.mlr.press/v48/linb16.pdf) ![](https://img.shields.io/badge/dataset-CIFAR--10-orange.svg)
- Proposes algorithm to convert floating point to fixed point 
- Converts CNN layer activations and weights into fixed point

#### Zhu, Chenzhuo, Song Han, Huizi Mao, and William J. Dally. "Trained ternary quantization." arXiv preprint arXiv:1612.01064 (2016). [link](https://arxiv.org/pdf/1612.01064.pdf)
