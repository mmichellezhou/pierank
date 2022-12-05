# pierank

## Install
 0. sudo apt install cmake libgflags-dev libgoogle-glog-dev pkg-config
 1. git clone https://github.com/mmichellezhou/pierank.git && cd pierank
 2. git submodule init && git submodule update
 3. mkdir release && cd release
 4. cmake -DCMAKE_BUILD_TYPE=Release ..
 5. make -j64