# PieRank: Embedded Large-Scale Sparse Matrix Processing

 
### License

PieRank is licensed under the MIT License.


## Introduction

I introduce *PieRank*, a library aimed at embedded large-scale sparse matrix processing. It enjoys significant advantages over previous state-of-the-art for handling big, sparse data sets in scalability, speed, and usability. My approach is also more general, allowing a wider variety of applications beyond graphs. Inspired by Raspberry Pi(e) for its low cost, I name the library  
PieRank  with the goal of dramatically reducing the hardware requirements  for cutting-edge sparse matrix processing. My experiments show PieRank outperforms <a href="https://github.com/GraphChi/graphchi-cpp/tree/master">GraphChi</a> by a wide margin, making it possible for a single embedded device to meet or exceed the scalability and performance of server-class computers on big matrices such as *AGATHA 2015*, a deep-learning network with 5.8B parameters as the largest instance in the popular SuiteSparse Matrix Collection.

Available as an open-source project on GitHub under the permissive MIT license, PieRank is implemented in plain C++ and compiles and runs on Linux, Mac OS, and Raspbian.


### Publication

PieRank is currently under review at the 2023 IEEE International Conference on Big Data. PDF of the paper can be viewed <a href="https://drive.google.com/file/d/12zmBz9jSg6DsNYV5vDceJvKcOidUlCcQ/view?usp=sharing">here</a>.


## Features

- Utilizes FlexArray, a late-binding random-access container for flexible data encoding. FlexArray supports high-speed mapping from a nominal integer type to a low-level binary representation that is more memory-efficient for the data.
- Employs asynchronous, multi-threaded parallel processing. Using a set of disjoint integer-based intervals called ranges, a user can write a PieRank Program (PRP) to process the matrix in parallel.
- Applies an efficient delta compression technique, called Sketch Compression, for (sorted) integer arrays with guaranteed worst-case constant O(1) decode time, an improvement of the conventional decoding techniques that often require multiple decode rounds to reach the target entry.
- Provides built-in support for memory-mappable sparse matrices for RAM-oblivious sparse matrix processing
- Offers a library for sparse matrix processing
- Contains sample applications <a href="https://github.com/mmichellezhou/pierank/blob/main/pierank/kernels/pagerank.h">PageRank</a> and <a href="https://github.com/mmichellezhou/pierank/blob/main/pierank/kernels/components.h">connected components</a>.
- Can run graphs with billions of edges, with linear scalability, on an embedded device such as a Raspberry Pi.
- Compiles and runs on Linux, Mac OS, and Raspbian.

## Getting Started

0.  sudo apt install cmake libgflags-dev libgoogle-glog-dev pkg-config
1.  git clone  [https://github.com/mmichellezhou/pierank.git](https://github.com/mmichellezhou/pierank.git)  && cd pierank
2.  git submodule init && git submodule update
3.  mkdir release && cd release
4.  cmake -DCMAKE_BUILD_TYPE=Release ..
5.  make -j64


## Performance

Table 1 is a comparison of encoding sizes, including matrix market file size (mtx size), PieRank matrix file size (prm size), compression ratio (ratio), optimal number of sketch bits (_s_*), number of bytes for each element of *COL_INDEX* (cb) and *ROW_INDEX* (rb) in PieRank’s binary <a href="https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)">CSC</a> format.

Table 1.
<table class="wikitable"><tr><td style="border: 1px solid #ccc; padding: 5px;"> <strong><i>p</i>#</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>mtx size</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>prm size</strong></td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>ratio</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong><i>s</i>*</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>cb</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>rb</strong></td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 1 </td><td style="border: 1px solid #ccc; padding: 5px;"> 30,579,901 </td><td style="border: 1px solid #ccc; padding: 5px;"> 7,502,539 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.1 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 11 </td><td style="border: 1px solid #ccc; padding: 5px;"> 2 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 2 </td><td style="border: 1px solid #ccc; padding: 5px;"> 70,636,197 </td><td style="border: 1px solid #ccc; padding: 5px;"> 16,946,668 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.2 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 13 </td><td style="border: 1px solid #ccc; padding: 5px;"> 2  </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td><td style="border: 1px solid #ccc; padding: 5px;"> 102,294,093 </td><td style="border: 1px solid #ccc; padding: 5px;"> 24,138,518 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.2 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 8 </td><td style="border: 1px solid #ccc; padding: 5px;"> 2 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 4 </td><td style="border: 1px solid #ccc; padding: 5px;"> 171,065,574 </td><td style="border: 1px solid #ccc; padding: 5px;"> 37,967,050 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.5 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 5 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 5 </td><td style="border: 1px solid #ccc; padding: 5px;"> 261,658,301 </td><td style="border: 1px solid #ccc; padding: 5px;"> 57,075,516 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.6 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 13 </td><td style="border: 1px solid #ccc; padding: 5px;"> 2 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 6 </td><td style="border: 1px solid #ccc; padding: 5px;"> 642,663,421 </td><td style="border: 1px solid #ccc; padding: 5px;"> 145,792,458 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.4 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 16 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 7 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1,011,609,587 </td><td style="border: 1px solid #ccc; padding: 5px;"> 217,282,551 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.7 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 6 </td><td style="border: 1px solid #ccc; padding: 5px;"> 2 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 8 </td><td style="border: 1px solid #ccc; padding: 5px;"> 16,451,551,195 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3,903,296,960 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.2 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 9 </td><td style="border: 1px solid #ccc; padding: 5px;"> 26,141,060,759 </td><td style="border: 1px solid #ccc; padding: 5px;"> 5,999,068,381 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.4 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 9 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 10 </td><td style="border: 1px solid #ccc; padding: 5px;"> 104,523,784,790 </td><td style="border: 1px solid #ccc; padding: 5px;"> 23,773,593,875 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.4 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 5 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4 </td></tr></table>

Table 2 is a comparison of matrix read time in seconds, including matrix market read time, PRM read time, speedup ratio, and PRM memory-map time.

Table 2.
<table class="wikitable"><tr><td style="border: 1px solid #ccc; padding: 5px;"> <strong><i>p</i>#</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>mtx size</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>prm size</strong></td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>ratio</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>prm mmap (sec)</strong> </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 1 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.41 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.16 </td><td style="border: 1px solid #ccc; padding: 5px;"> 8.8 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.0006 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 2 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3.52 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.17 </td><td style="border: 1px solid #ccc; padding: 5px;"> 20.4 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.0024 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.93 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.14 </td><td style="border: 1px solid #ccc; padding: 5px;"> 34.3 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.0027 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 4 </td><td style="border: 1px solid #ccc; padding: 5px;"> 7.06 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.60 </td><td style="border: 1px solid #ccc; padding: 5px;"> 11.7 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.0090 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 5 </td><td style="border: 1px solid #ccc; padding: 5px;"> 10.61 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.43 </td><td style="border: 1px solid #ccc; padding: 5px;"> 7.4 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.0027 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 6 </td><td style="border: 1px solid #ccc; padding: 5px;"> 28.29 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3.97 </td><td style="border: 1px solid #ccc; padding: 5px;"> 7.1 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.0028 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 7 </td><td style="border: 1px solid #ccc; padding: 5px;"> 52.75 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3.91 </td><td style="border: 1px solid #ccc; padding: 5px;"> 13.5 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.0062 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 8 </td><td style="border: 1px solid #ccc; padding: 5px;"> 721.53 </td><td style="border: 1px solid #ccc; padding: 5px;"> 17.5 </td><td style="border: 1px solid #ccc; padding: 5px;"> 41.2 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.0025 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 9 </td><td style="border: 1px solid #ccc; padding: 5px;"> 896.75 </td><td style="border: 1px solid #ccc; padding: 5px;"> 83.93 </td><td style="border: 1px solid #ccc; padding: 5px;"> 10.7 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.0077 </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 10 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3,524.49 </td><td style="border: 1px solid #ccc; padding: 5px;"> 727.82 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.8 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.1309 </td></tr></table>

Table 3 is a comparison of average PageRank iteration time, including GraphChi (*GC*), in-RAM PieRank (*PR<sub>RAM</sub>*), RAM-oblivious PieRank (*PR<sub>RO</sub>*), and speedup ratio for *PR<sub>RO</sub>*.

Table 3.
<table class="wikitable"><tr><td style="border: 1px solid #ccc; padding: 5px;"> <strong>p#</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong><i>GC</i> (sec)</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong><i>PR<sub>RAM</sub></i> (sec)</strong></td><td style="border: 1px solid #ccc; padding: 5px;"> <strong><i>PR<sub>RO</sub></i> (sec)</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>ratio</strong> </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 1 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.64 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.06 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.06 </td><td style="border: 1px solid #ccc; padding: 5px;"> 10.7 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 2 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.23 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.06 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.07 </td><td style="border: 1px solid #ccc; padding: 5px;"> 17.6 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.20 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.05 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.05 </td><td style="border: 1px solid #ccc; padding: 5px;"> 24.0 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 4 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3.74 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.21 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.22 </td><td style="border: 1px solid #ccc; padding: 5px;"> 17.0 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 5 </td><td style="border: 1px solid #ccc; padding: 5px;"> 6.10 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.52 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.52 </td><td style="border: 1px solid #ccc; padding: 5px;"> 11.7 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 6 </td><td style="border: 1px solid #ccc; padding: 5px;"> 12.04 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.42 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.43 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 8.4 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 7 </td><td style="border: 1px solid #ccc; padding: 5px;"> 15.42 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.57 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.59 </td><td style="border: 1px solid #ccc; padding: 5px;"> 9.7 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 8 </td><td style="border: 1px solid #ccc; padding: 5px;"> 152.03 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3.62 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.34 </td><td style="border: 1px solid #ccc; padding: 5px;"> 35.0 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 9 </td><td style="border: 1px solid #ccc; padding: 5px;"> 386.75 </td><td style="border: 1px solid #ccc; padding: 5px;"> 44.47 </td><td style="border: 1px solid #ccc; padding: 5px;"> 44.60 </td><td style="border: 1px solid #ccc; padding: 5px;"> 8.7 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 10 </td><td style="border: 1px solid #ccc; padding: 5px;"> 2,207.72 </td><td style="border: 1px solid #ccc; padding: 5px;"> MEM </td><td style="border: 1px solid #ccc; padding: 5px;"> 328.11 </td><td style="border: 1px solid #ccc; padding: 5px;"> 6.7 x </td></tr></table>

Table 4 is a comparison of average connected-components iteration time, including GraphChi (*GC*), in-RAM PieRank (*PR<sub>RAM</sub>*), RAM-oblivious PieRank (*PR<sub>RO</sub>*), and speedup ratio for *PR<sub>RO</sub>*.

Table 4.
<table class="wikitable"><tr><td style="border: 1px solid #ccc; padding: 5px;"> <strong>p#</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong><i>GC</i> (sec)</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong><i>PR<sub>RAM</sub></i> (sec)</strong></td><td style="border: 1px solid #ccc; padding: 5px;"> <strong><i>PR<sub>RO</sub></i> (sec)</strong> </td><td style="border: 1px solid #ccc; padding: 5px;"> <strong>ratio</strong> </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 1 </td><td style="border: 1px solid #ccc; padding: 5px;"> 2.19 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.06 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.06 </td><td style="border: 1px solid #ccc; padding: 5px;"> 36.5 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 2 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3.29 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.06 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.07 </td><td style="border: 1px solid #ccc; padding: 5px;"> 47.0 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 3 </td><td style="border: 1px solid #ccc; padding: 5px;"> 3.88 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.05 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.05 </td><td style="border: 1px solid #ccc; padding: 5px;"> 77.6 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 4 </td><td style="border: 1px solid #ccc; padding: 5px;"> 14.22 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.25 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.25 </td><td style="border: 1px solid #ccc; padding: 5px;"> 56.9 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 5 </td><td style="border: 1px solid #ccc; padding: 5px;"> 12.60 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.60 </td><td style="border: 1px solid #ccc; padding: 5px;"> 0.61 </td><td style="border: 1px solid #ccc; padding: 5px;"> 20.7 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 6 </td><td style="border: 1px solid #ccc; padding: 5px;"> 27.54 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.57 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.62 x </td><td style="border: 1px solid #ccc; padding: 5px;"> 17.0 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 7 </td><td style="border: 1px solid #ccc; padding: 5px;"> 35.43 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.49 </td><td style="border: 1px solid #ccc; padding: 5px;"> 1.58 </td><td style="border: 1px solid #ccc; padding: 5px;"> 22.4 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 8 </td><td style="border: 1px solid #ccc; padding: 5px;"> 195.48 </td><td style="border: 1px solid #ccc; padding: 5px;"> 4.36 </td><td style="border: 1px solid #ccc; padding: 5px;"> 6.12 </td><td style="border: 1px solid #ccc; padding: 5px;"> 31.9 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 9 </td><td style="border: 1px solid #ccc; padding: 5px;"> 812.55 </td><td style="border: 1px solid #ccc; padding: 5px;"> 47.09 </td><td style="border: 1px solid #ccc; padding: 5px;"> 46.50 </td><td style="border: 1px solid #ccc; padding: 5px;"> 17.5 x </td></tr> <tr><td style="border: 1px solid #ccc; padding: 5px;"> 10 </td><td style="border: 1px solid #ccc; padding: 5px;"> MEM </td><td style="border: 1px solid #ccc; padding: 5px;"> MEM </td><td style="border: 1px solid #ccc; padding: 5px;"> 350.06 </td><td style="border: 1px solid #ccc; padding: 5px;"> N/A </td></tr></table>

## How to Reproduce Our Experimental Results

Please help yourself to the <a href="https://github.com/mmichellezhou/pierank/tree/main/scripts">scripts  directory</a> which contains all of the Bash scripts I use to run PieRank, including <a href="https://github.com/mmichellezhou/pierank/blob/main/scripts/pagerank.sh">PageRank</a> and <a href="https://github.com/mmichellezhou/pierank/blob/main/scripts/components.sh">connected components</a>.
