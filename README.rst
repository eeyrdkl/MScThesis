MSc Computer Science and Engineering
====================================

Thesis Implementation

Index
-----
- `Requirements`_
- `Installation`_
- `Post-Installation`_

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
$ julia

julia> Pkg.clone("https://github.com/ekyurdakul/MScThesis.git");

Post-Installation
-----------------
Download preprocessed data into /data/

Download pretrained weights into /data/pretrained/