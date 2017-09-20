MSc Computer Science and Engineering Thesis Implementation
==========================================================

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
  julia> Pkg.clone("https://github.com/ekyurdakul/MScThesis.git");

Post-Installation
-----------------
Download preprocessed data into /data/

Download pretrained weights into /data/pretrained/

Citation
--------
If you benefit from our research, please cite our paper:

::

  @inproceedings{Yurdakul2017,
    author = {Yurdakul, Ekrem Emre and Yemez, Y\"{u}cel},
    title = {{Semantic Segmentation of RGBD Videos with Recurrent Fully Convolutional Neural Networks}},
    booktitle = {ICCV 2017 4th IEEE/ISPRS Joint Workshop on Multi-Sensor Fusion for Dynamic Scene Understanding},
    year = {2017}
  }

If you use the preprocessed data, please cite the corresponding datasets:

- Virtual KITTI

::

  @inproceedings{Gaidon:Virtual:CVPR2016,
    author = {Gaidon, A and Wang, Q and Cabon, Y and Vig, E},
    title = {Virtual Worlds as Proxy for Multi-Object Tracking Analysis},
    booktitle = {CVPR},
    year = {2016}
  }

- DAVIS

::

  @inproceedings{Perazzi2016,
    author = {F. Perazzi and J. Pont-Tuset and B. McWilliams and L. {Van Gool} and M. Gross and A. Sorkine-Hornung},
    title = {A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation},
    booktitle = {Computer Vision and Pattern Recognition},
    year = {2016}
  }

- Robot\@Home

::

  @article{Ruiz-Sarmiento-IJRR-2017,
    author = {Ruiz-Sarmiento, J. R. and Galindo, Cipriano and Gonz{\'{a}}lez-Jim{\'{e}}nez, Javier},
    title = {Robot@Home, a Robotic Dataset for Semantic Mapping of Home Environments},
    journal = {International Journal of Robotics Research},
    year = {2017}
  }
