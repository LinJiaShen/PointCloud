# Install deep global registration

## system reqiurements
    Ubuntu 14.04 or higher
    CUDA 10.1.243 or higher
    pytorch 1.5 or higher
    python 3.6 or higher
    GCC 7

install the MinkowskiEngine
```
# Install MinkowskiEngine
sudo apt install libopenblas-dev g++-7
pip install torch
export CXX=g++-7; pip install -U MinkowskiEngine --install-option="--blas=openblas" -v

# Download and setup DeepGlobalRegistration
git clone https://github.com/chrischoy/DeepGlobalRegistration.git
cd DeepGlobalRegistration
pip install -r requirements.txt
```


## Download model
model zoo[https://github.com/chrischoy/DeepGlobalRegistration#demo]


## credit
deep global registration[https://github.com/chrischoy/DeepGlobalRegistration]
