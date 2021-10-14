# G-SLIDE

G-SLIDE is a GPU-based sub-linear deep learning engine via LSH sparsification of fully-connected neural networks. The details can be found in this [paper]().

## Dataset

The Datasets can be downloaded in [Amazon-670K](https://drive.google.com/open?id=0B3lPMIHmG6vGdUJwRzltS1dvUVk).

## Baselines

The baseline is [SLIDE](https://github.com/keroro824/HashingDeepLearning). The source codes of TensorFlow-CPU and TensorFlow-GPU baselines can also be found from the same link.

## Running G-SLIDE

### Environments

The experiment environments in the paper are as follow:

* OS: Ubuntu 20.04
* Compiler: nvcc 11.1
* GPU: 2080ti
* CPU: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz

### Dependencies

* cuBLAS
* Thrust
* [JsonCpp](https://github.com/open-source-parsers/jsoncpp): we use it to parse the configuration json file.

### Compile and Run

We've provided the Makefile so to compile the project:

```bash
git clone https://github.com/PanZaifeng/G-SLIDE.git
cd G-SLIDE
make
```

 Before running of G-SLIDE, you should download the dataset of Amazon-670K and configure the `amazon.json` properly. Then to run it:

```bash
./runme ./amazon.json
```



