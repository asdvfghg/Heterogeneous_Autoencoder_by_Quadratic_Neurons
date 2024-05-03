# Quadratic Neuron-empowered Heterogeneous Autoencoder for Unsupervised Anomaly Detection
This is the repository of the paper "Quadratic Neuron-empowered Heterogeneous Autoencoder for Unsupervised Anomaly Detection", published in IEEE Transactions on Artificial Intelligence
 [Paper](https://ieeexplore.ieee.org/document/10510400)
## Abstract
Inspired by the complexity and diversity of biological neurons, a quadratic neuron is proposed to replace the inner product in the current neuron with a simplified quadratic function. Employing such a novel type of neurons offers a new perspective on developing deep learning. When analyzing quadratic neurons, we find that there exists a function such that a heterogeneous network can approximate it well with a polynomial number of neurons but a purely conventional or quadratic network needs an exponential number of neurons to achieve the same level of error. Encouraged by this inspiring theoretical result on heterogeneous networks, we directly integrate conventional and quadratic neurons in an autoencoder to make a new type of heterogeneous autoencoders. To our best knowledge, it is the first heterogeneous autoencoder that is made of different types of neurons. Next, we apply the proposed heterogeneous autoencoder
to unsupervised anomaly detection for tabular data and bearing fault signals. The anomaly detection faces difficulties such as data unknownness, anomaly feature heterogeneity, and feature unnoticeability, which is suitable for the proposed heterogeneous autoencoder. Its high feature representation ability can characterize a variety of anomaly data (heterogeneity), discriminate the anomaly from the normal (unnoticeability), and accurately learn the distribution of normal samples (unknownness). Experiments show that heterogeneous autoencoders perform competitively compared to other state-of-the-art models.

## Citing
If you find this repo useful for your research, please consider citing it:
```
@ARTICLE{10510400,
  author={Liao, Jing-Xiao and Hou, Bo-Jian and Dong, Hang-Cheng and Zhang, Hao and Zhang, Xiaoge and Sun, Jinwei and Zhang, Shiping and Fan, Feng-Lei},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={Quadratic Neuron-empowered Heterogeneous Autoencoder for Unsupervised Anomaly Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Neurons;Anomaly detection;Heterogeneous networks;Task analysis;Biological neural networks;Deep learning;Biological system modeling;Deep learning theory;heterogeneous autoencoder;quadratic neuron;anomaly detection},
  doi={10.1109/TAI.2024.3394795}}
```



## Heterogeneous Autoencoder

### Quadratic Neurons
A quadratic neuron was proposed by [1], It computes two inner products  and  one  power  term  of  the  input  vector  and  integrates them for a nonlinear activation function. The output function of a quadratic neuron is expressed as 

![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/小书匠/1641001696385.png),

where $\sigma(\cdot)$ is a nonlinear activation function, $\odot$ denotes the Hadamard product, $\boldsymbol{w}^r,\boldsymbol{w}^g, \boldsymbol{w}^b\in\mathbb{R}^n$ are weight vectors, and $b^r, b^g, c\in\mathbb{R}$ are biases. When $\boldsymbol{w}^g=0$, $b^g=1$, and $\boldsymbol{w}^b=0$, a quadratic neuron degenerates to a conventional neuron:  $\sigma(f(\boldsymbol{x}))= \sigma(\boldsymbol{x}^\top\boldsymbol{w}^{r}+b^{r})$. 

### HAE Structure
We propose three heterogeneous autoencoders integrating conventional  and  quadratic  neurons  in  one  model,  referred  to  as HAE-X, HAE-Y, and HAE-I, respectively.
![The scheme of HAE-X, HAE-Y, and HAE-I.](https://raw.githubusercontent.com/asdvfghg/image/master/小书匠/1641001696444.png)

## Repository organization

### Requirements
We use PyCharm 2021.2 to be a coding IDE, if you use the same, you can run this program directly. Other IDE we have not yet tested, maybe you need to change some settings.
* Python == 3.8
* PyTorch == 1.10.1
* CUDA == 11.3 if use GPU
* pyod == 0.9.6
* anaconda == 2021.05
 
### Organization
```
HAE_Empowered_by_Quadratic_Neurons
│   benchmark_method.py # Implementation of OCSVM, SUDO, DeepSVDD
│   train_ae.py # Implementation of AE\QAE\HAEs 
│   train_DAGMM.py # Implementation of DAGMM
│   train_RCA.py # Implementation of RCA
└─  data # Anomaly detection datasets 
└─  utils
     │   data_process.py # data_process functions for RCA and DAGMM
     │   QuadraticOperation.py # Quadratic neuron function
     │   train_function.py # Train function for quadratic network
└─  Model
     │   DAGMM.py 
     │   HAutoEncoder.py # Contain AE\QAE\HAEs 
     │   RCA.py 

```

### Datasets
We use the ODDs dataset[2]. More details can be found in [Official Page of ODDs Dataset](http://odds.cs.stonybrook.edu).

### How to Use

Run ```train_ae.py``` to train an autoencoder. We provide three heterogeneous autoencoders, a quadratic and a conventional autoencoder. 

Run ```benchmark_method.py``` to train  OCSVM, SUDO, DeepSVDD. We follow the implementation by [pyod](https://github.com/yzhao062/pyod) package [3].
 
 Run ```train_DAGMM.py``` to train  DAGMM[4]. We follow the implementation by [RCA](https://github.com/illidanlab/RCA).

 Run ```train_RCA.py``` to train  RCA[5]. We follow the implementation by [RCA](https://github.com/illidanlab/RCA).

All results will be saved to the ***'results'*** folder.

## Contact
If you have any questions about our work, please contact the following email address:

jingxiaoliao@hit.edu.cn

Enjoy your coding!
## Reference
[1] Fenglei Fan, Wenxiang Cong, and Ge Wang. A new type of neurons for machine learning. International journal for numericalmethods in biomedical engineering, 34(2):e2920, 2018.

[2] Shebuti Rayana.  Odds library [http://odds.cs.stonybrook.edu]. stony brook, ny:  Stony brook university.Department of ComputerScience, 2016.

[3] Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.

[4]  B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, and H. Chen, “Deep autoencoding gaussian mixture model for unsupervised anomaly detection,” in International conference on learning representations, 2018.

[5] B. Liu, D. Wang, K. Lin, P.-N. Tan, and J. Zhou, “RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection,” pp. 1505–1511, 2021, doi: 10.24963/ijcai.2021/208.
