# PieRank: Embedded Large-Scale Sparse Matrix Processing

 
### License

PieRank is licensed under the MIT License.


## Introduction

I introduce *PieRank*, a library aimed at embedded large-scale sparse matrix processing. It enjoys significant advantages over previous state-of-the-art for handling big, sparse data sets in scalability, speed, and usability. My approach is also more general, allowing a wider variety of applications beyond graphs. Inspired by [Raspberry Pi(e)](https://www.raspberrypi.org/) for its low cost, I name the library PieRank with the goal of dramatically reducing the hardware requirements  for cutting-edge sparse matrix processing. My experiments show PieRank outperforms [GraphChi](https://github.com/GraphChi/graphchi-cpp) by a wide margin, making it possible for a single embedded device to meet or exceed the scalability and performance of server-class computers on big matrices such as [AGATHA 2015](https://sparse.tamu.edu/Sybrandt/AGATHA_2015), a deep-learning network with 5.8B parameters as the largest instance in the popular [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

Available as an open-source project on [GitHub](https://github.com/mmichellezhou/pierank) under the permissive MIT license, PieRank is implemented in plain C++ and compiles and runs on Linux, Mac OS, and [Raspbian](http://www.raspbian.org/).

### Publication

PieRank is currently under review at the 2023 IEEE International Conference on Big Data. PDF of the paper can be viewed <a href="https://drive.google.com/file/d/12zmBz9jSg6DsNYV5vDceJvKcOidUlCcQ/view?usp=sharing">here</a>.


## Features

- Utilizes FlexArray, a late-binding random-access container for flexible data encoding. FlexArray supports high-speed mapping from a nominal integer type to a low-level binary representation that is more memory-efficient for the data.
- Employs asynchronous, multi-threaded parallel processing. Using a set of disjoint integer-based intervals called ranges, a user can write a PieRank Program (PRP) to process the matrix in parallel.
- Applies an efficient delta compression technique, called Sketch Compression, for (sorted) integer arrays with guaranteed worst-case constant O(1) decode time, an improvement of the conventional decoding techniques that often require multiple decode rounds to reach the target entry.
- Provides built-in support for memory-mappable sparse matrices for RAM-oblivious sparse matrix processing
- Offers a library for sparse matrix processing
- Contains sample applications [PageRank](https://github.com/mmichellezhou/pierank/blob/main/pierank/kernels/pagerank.h) and [connected components](https://github.com/mmichellezhou/pierank/blob/main/pierank/kernels/components.h).
- Can run graphs with billions of edges, with linear scalability, on an embedded device such as a [Raspberry Pi](https://www.raspberrypi.org/).
- Compiles and runs on Linux, Mac OS, and [Raspbian](http://www.raspbian.org/).

## Getting Started

0.  sudo apt install cmake libgflags-dev libgoogle-glog-dev pkg-config
1.  git clone  [https://github.com/mmichellezhou/pierank.git](https://github.com/mmichellezhou/pierank.git)  && cd pierank
2.  git submodule init && git submodule update
3.  mkdir release && cd release
4.  cmake -DCMAKE_BUILD_TYPE=Release ..
5.  make -j64


## Experimental Results

For all experiments, I used a [Raspberry Pi 4 Model B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) with 8GB of RAM and a 960GB ADATA SC680 USB 3.2 Gen 2 external SSD with a maximum read (write) speed of 530 (460) MB/s. The OS is 64-bit Raspbian [Buster](https://www.raspberrypi.com/news/buster-the-new-version-of-raspbian/). GCC 8.3.0 was used to generate all binaries, including [GraphChi](https://github.com/GraphChi/graphchi-cpp).

| $p$\# | name               | \#rows   | nnz           |
|-|-|-|-|
| 1   | [web-Stanford](https://sparse.tamu.edu/SNAP/web-Stanford)       | 281,903     | 2,312,497     |
| 2   | [amazon-2008](https://sparse.tamu.edu/LAW/amazon-2008)      | 735,323     | 5,158,388     |
| 3   | [Stanford\_Berkeley](https://sparse.tamu.edu/Kamvar/Stanford_Berkeley) | 683,446     | 7,583,376     |
| 4   | [333SP](https://sparse.tamu.edu/DIMACS10/333SP)              | 3,712,815   | 11,108,633    |
| 5   | [cit-Patents](https://sparse.tamu.edu/SNAP/cit-Patents)        | 3,774,768   | 16,518,948    |
| 6   | [wikipedia-20070206](https://sparse.tamu.edu/Gleich/wikipedia-20070206) | 3,566,907   | 45,030,389    |
| 7   | [soc-LiveJournal1](https://sparse.tamu.edu/SNAP/soc-LiveJournal1)   | 4,847,571   | 68,993,773    |
| 8   | [uk-2005](https://sparse.tamu.edu/LAW/uk-2005)            | 39,459,925  | 936,364,282   |
| 9   | [twitter-2010](https://sparse.tamu.edu/SNAP/twitter7)       | 41,652,230  | 1,468,365,182 |
| 10  | [AGATHA\_2015](https://sparse.tamu.edu/Sybrandt/AGATHA_2015)       | 183,964,077 | 5,794,362,982 |

The above table shows the sparse matrices used in the experiments. They are from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) website, including [GATHA 2015](https://sparse.tamu.edu/Sybrandt/AGATHA_2015) as the largest instance in the collection. It is a deep-learning network with 5.8B parameters (or 11.6B if one counts each undirected edge as two parameters). Instead of referring to matrices by their names, the rest of the tables all use their problem number mentioned above (e.g., $p$\#10 refers to ``AGATHA\_2015").

| $p$\# | mtx size            | prm size | ratio | $s^*$ | cb | rb |
|-|-|-|-|-|-|-|
| 1   | 30,579,901      | 7,502,539              | 4.1$\times$           | 11                | 2                | 3              |
| 2   | 70,636,197      | 16,946,668             | 4.2$\times$            | 13                | 2                | 3              |
| 3   | 102,294,093     | 24,138,518             | 4.2$\times$            | 8                 | 2                | 3              |
| 4   | 171,065,574     | 37,967,050             | 4.5$\times$            | 5                 | 1                | 3              |
| 5   | 261,658,301     | 57,075,516             | 4.6$\times$            | 13                | 2                | 3              |
| 6   | 642,663,421     | 145,792,458            | 4.4$\times$            | 16                | 3                | 3              |
| 7   | 1,011,609,587   | 217,282,551            | 4.7$\times$            | 6                 | 2                | 3              |
| 8   | 16,451,551,195  | 3,903,296,960          | 4.2$\times$            | 0                 | 4                | 4              |
| 9   | 26,141,060,759  | 5,999,068,381          | 4.4$\times$            | 9                 | 3                | 4              |
| 10  | 104,523,784,790 | 23,773,593,875         | 4.4$\times$            | 5                 | 3                | 4              |

The table above is a comparison of encoding sizes, including [matrix market](https://math.nist.gov/MatrixMarket/formats.html) file size (mtx size), PieRank matrix file size (prm size), compression ratio (ratio), optimal number of sketch bits (_s_*), number of bytes for each element of *COL_INDEX* (cb) and *ROW_INDEX* (rb) in PieRank’s binary [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)) format.

| $p$\#   |  mtx read (sec) |  prm read (sec) |  ratio | prm mmap (sec) |
|-|-|-|-|-|
| 1  | 1.41     | 0.16     | 8.8$\times$  | 0.0006   |
| 2  | 3.52     | 0.17     | 20.4$\times$ | 0.0024   |
| 3  | 4.93     | 0.14     | 34.3$\times$ | 0.0027   |
| 4  | 7.06     | 0.60     | 11.7$\times$ | 0.0090   |
| 5  | 10.61    | 1.43     | 7.4$\times$  | 0.0027   |
| 6  | 28.29    | 3.97     | 7.1$\times$  | 0.0028   |
| 7  | 52.75    | 3.91     | 13.5$\times$ | 0.0062   |
| 8  | 721.53   | 17.50    | 41.2$\times$ | 0.0025   |
| 9  | 896.75   | 83.93    | 10.7$\times$ | 0.0077   |
| 10 | 3,524.49 | 727.82   | 4.8$\times$  | 0.1309   |

The above table is a comparison of matrix read time in seconds, including matrix market read time, PRM read time, speedup ratio, and PRM [memory-map](https://en.wikipedia.org/wiki/Memory-mapped_file) time.

| $p$\# | $GC$ (sec) | $PR_\textrm{RAM}$ (sec) | $PR_\textrm{RO}$ (sec) | ratio |
|-|-|-|-|-|
| 1   | 0.64          | 0.06        | 0.06       | 10.7$\times$ |
| 2   | 1.23          | 0.06        | 0.07       | 17.6$\times$ |
| 3   | 1.20          | 0.05        | 0.05       | 24.0$\times$ |
| 4   | 3.74          | 0.21        | 0.22       | 17.0$\times$ |
| 5   | 6.10          | 0.52        | 0.52       | 11.7$\times$ |
| 6   | 12.04         | 1.42        | 1.43       | 8.4$\times$  |
| 7   | 15.42         | 1.57        | 1.59       | 9.7$\times$  |
| 8   | 152.03        | 3.62        | 4.34       | 35.0$\times$ |
| 9   | 386.75        | 44.47       | 44.60      | 8.7$\times$  |
| 10  | 2,207.72      | MEM         | 328.11     | 6.7$\times$  |


The table above is a comparison of average [PageRank](https://en.wikipedia.org/wiki/PageRank) iteration time, including [GraphChi](https://github.com/GraphChi/graphchi-cpp) (*GC*), in-RAM PieRank (*PR<sub>RAM</sub>*), [RAM-oblivious](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm) PieRank (*PR<sub>RO</sub>*), and speedup ratio for *PR<sub>RO</sub>*.

| $p$\# | $GC$ (sec) | $PR_\textrm{RAM}$ (sec) | $PR_\textrm{RO}$ (sec) | ratio |
|-|-|-|-|-|
| 1   | 2.19          | 0.06        | 0.06       | 36.5$\times$ |
| 2   | 3.29          | 0.06        | 0.07       | 47.0$\times$ |
| 3   | 3.88          | 0.05        | 0.05       | 77.6$\times$ |
| 4   | 14.22         | 0.25        | 0.25       | 56.9$\times$ |
| 5   | 12.60         | 0.60        | 0.61       | 20.7$\times$ |
| 6   | 27.54         | 1.57        | 1.62       | 17.0$\times$ |
| 7   | 35.43         | 1.49        | 1.58       | 22.4$\times$ |
| 8   | 195.48        | 4.36        | 6.12       | 31.9$\times$ |
| 9   | 812.55        | 47.09       | 46.50      | 17.5$\times$ |
| 10  | MEM           | MEM         | 350.06     | N/A   |

The above table is a comparison of average [connected-components](https://en.wikipedia.org/wiki/Connected-component_labeling) iteration time, including [GraphChi](https://github.com/GraphChi/graphchi-cpp) (*GC*), in-RAM PieRank (*PR<sub>RAM</sub>*), [RAM-oblivious](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm) PieRank (*PR<sub>RO</sub>*), and speedup ratio for *PR<sub>RO</sub>*.

## How to Reproduce Experimental Results

Please help yourself to the [scripts  directory](https://github.com/mmichellezhou/pierank/tree/main/scripts) which contains all of the Bash scripts I use to run PieRank, including:
- [mtx_to_prm.sh](https://github.com/mmichellezhou/pierank/blob/main/scripts/mtx_to_prm.sh) for matrix-market-to-PRM conversion
- [prm_print.sh](https://github.com/mmichellezhou/pierank/blob/main/scripts/prm_print.sh) for pretty-printing of PRM files
- [pagerank.sh](https://github.com/mmichellezhou/pierank/blob/main/scripts/pagerank.sh) for PageRank
- [components.sh](https://github.com/mmichellezhou/pierank/blob/main/scripts/components.sh) for Connected Components
