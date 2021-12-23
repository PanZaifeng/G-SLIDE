# G-SLIDE

G-SLIDE is a GPU-based sub-linear deep learning engine via LSH sparsification of fully-connected neural networks. The details can be found in this [paper](https://ieeexplore.ieee.org/document/9635657).

## Dataset

The Datasets can be downloaded in [Amazon-670K](https://drive.google.com/open?id=0B3lPMIHmG6vGdUJwRzltS1dvUVk) and [WikiLSHTC-325K](https://drive.google.com/file/d/0B3lPMIHmG6vGSHE1SWx4TVRva3c/view?resourcekey=0-ZGNqdLuqttRdnAj-U0bktA).

## Baselines

The baseline is [SLIDE](https://github.com/keroro824/HashingDeepLearning). The source codes of TensorFlow-CPU and TensorFlow-GPU baselines can also be found from the same link.

## Running G-SLIDE

### Tested Environments

The experiment environments in the paper are as follow:

* OS: Ubuntu 20.04
* Compiler: nvcc 11.1
* GPU: 2080ti
* CPU: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
* CMake：3.14 and above

### Dependencies

* cuBLAS
* Thrust
* [JsonCpp](https://github.com/open-source-parsers/jsoncpp): we use it to parse the configuration json file.

### Compile and Run

Type the following commands to compile the project:

```bash
git clone https://github.com/PanZaifeng/G-SLIDE.git
cd G-SLIDE
cmake -B build
cmake --build build
```

Before running G-SLIDE, you should download the **dataset** of Amazon-670K and re-configure the `amazon.json` properly. Note that there will be lots of information to be printed, so we recommend **redirecting stdout** when running.

```bash
./runme ./amazon.json > amazon.log
```
