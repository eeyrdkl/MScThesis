MSc Computer Science and Engineering - Thesis Implementation
==========================================================

Title: Semantic Segmentation of RGBD Videos with Recurrent Fully Convolutional Neural Networks

Index
-----
- `Requirements`_
- `Installation`_
- `Post-Installation`_
- `Citation`_

Requirements
------------
- Julia v0.5.2
- Julia Package: Knet
- Julia Package: AutoGrad
- Julia Package: JLD
- Julia Package: Images v0.5.14
- Julia Package: Colors
- NVIDIA K80 12 GB (or equivalent)
- Lots of RAM (>= 80 GB)

Installation
------------
::

  $ julia
  julia> Pkg.clone("https://github.com/eeyrdkl/MScThesis.git");

Post-Installation
-----------------
Download and unzip the preprocessed data into /data/

- Virtual KITTI: `Download [7.8 GB] <https://drive.google.com/file/d/0BzsWerNms8SNZFdkSDNzVHMycnc/view?usp=sharing>`_
- DAVIS: `Download [1.2 GB] <https://drive.google.com/file/d/0BzsWerNms8SNOFFuaV82akJmVjA/view?usp=sharing>`_
- Robot\@Home: `Download [3.2 GB] <https://drive.google.com/file/d/0BzsWerNms8SNcEVYTDJFMXMxZzQ/view?usp=sharing>`_

Download and unzip the pretrained weights into /data/pretrained/

- Pretrained Weights: `Download [1.6 GB] <https://drive.google.com/file/d/0BzsWerNms8SNaFdBWktsVGgweWM/view?usp=sharing>`_

Sample output images: `Download [44 MB] <https://drive.google.com/file/d/0BzsWerNms8SNaVN2UnU3bHFRdVU/view?usp=sharing>`_

Citation
--------
If you benefit from our research, please cite our paper:

::

  @inproceedings{Yurdakul_2017_ICCV,
    author = {Emre Yurdakul, Ekrem and Yemez, Yucel},
    title = {Semantic Segmentation of RGBD Videos With Recurrent Fully Convolutional Neural Networks},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
  }

If you use the preprocessed data, please cite the corresponding datasets:

- Virtual KITTI `[LICENSE] <http://www.europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds>`_

::

  @inproceedings{Gaidon:Virtual:CVPR2016,
    author = {Gaidon, A and Wang, Q and Cabon, Y and Vig, E},
    title = {Virtual Worlds as Proxy for Multi-Object Tracking Analysis},
    booktitle = {CVPR},
    year = {2016}
  }

- DAVIS `[LICENSE] <http://davischallenge.org/>`_

::

  @inproceedings{Perazzi2016,
    author = {F. Perazzi and J. Pont-Tuset and B. McWilliams and L. {Van Gool} and M. Gross and A. Sorkine-Hornung},
    title = {A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation},
    booktitle = {Computer Vision and Pattern Recognition},
    year = {2016}
  }

- Robot\@Home `[LICENSE] <http://mapir.isa.uma.es/mapirwebsite/index.php/mapir-downloads/203-robot-at-home-dataset.html>`_

::

  @article{Ruiz-Sarmiento-IJRR-2017,
    author = {Ruiz-Sarmiento, J. R. and Galindo, Cipriano and Gonz{\'{a}}lez-Jim{\'{e}}nez, Javier},
    title = {Robot@Home, a Robotic Dataset for Semantic Mapping of Home Environments},
    journal = {International Journal of Robotics Research},
    year = {2017}
  }

If you use the pretrained weights, please cite the VGG paper: `[LICENSE] <http://www.robots.ox.ac.uk/~vgg/research/very_deep/>`_

::

  @article{Simonyan14c,
    author = "Simonyan, K. and Zisserman, A.",
    title = "Very Deep Convolutional Networks for Large-Scale Image Recognition",
    journal = "CoRR",
    volume = "abs/1409.1556",
    year = "2014"
  }
