# tensorflow-MNIST-cGAN
Tensorflow implementation of condition Generative Adversarial Networks (cGAN) [1] for MANIST [2] dataset.

* you can download
  - MNIST dataset: http://yann.lecun.com/exdb/mnist/
 
## Resutls
* Generate using fixed noise (fixed_z_)

<table align='center'>
<tr align='center'>
<td> cGAN</td>
</tr>
<tr>
<td><img src = 'MNIST_cGAN_results/MNIST_cGAN_generation_animation.gif'>
</tr>
</table>

* MNIST vs Generated images

<table align='center'>
<tr align='center'>
<td> MNIST </td>
<td> cGAN after 100 epochs </td>
</tr>
<tr>
<td><img src = 'MNIST_cGAN_results/raw_MNIST.png'>
<td><img src = 'MNIST_cGAN_results/MNIST_cGAN_100.png'>
</tr>
</table>

* Training loss
  * cGAN

![Loss](MNIST_cGAN_results/MNIST_cGAN_train_hist.png)

* Learning time
    * MNIST cGAN - Avg. per epoch: 3.21 sec; Total 100 epochs: 1800.37 sec

## Development Environment

* Windows 7
* GTX1080 ti
* cuda 8.0
* Python 3.5.3
* tensorflow-gpu 1.2.1
* numpy 1.13.1
* matplotlib 2.0.2
* imageio 2.2.0

## Reference

[1] Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).

(Full paper: https://arxiv.org/pdf/1411.1784.pdf)

[3] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
