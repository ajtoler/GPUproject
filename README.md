All profile reports are located in "profileReports"

To set-up:
```
ssh g38nXX # XX:01-16
export LD_LIBRARY_PATH=/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin
make
```

To run:
```
./project_cpu
./project_cuda
./project_cuda_managed
./project_cuda_shared
./project_cpu_alpaka
```