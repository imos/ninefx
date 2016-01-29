# ninefx
[Private] ないんたんの為替予測プログラム


# EC2 サーバーのセットアップ

```
sudo yum install -y gcc48-c++.x86_64
wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
tar xzvf glog-0.3.3.tar.gz
cd glog-0.3.3
./configure
sudo make -j install
sudo ldconfig
```
