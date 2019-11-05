# moltr
Multi-objective learning to rank with LightGBM

The repository contains

* a python package (moltr) including a cython extension,
* a jupyter notebook with an example of using the package to perform learning to rank with multiple objectives. 

The cython extension comes prebuilt (on macOS). To rebuild it from sources, run

```
cd moltr; ./build.sh
```

Note: the extension uses OpenMP. To rebuild it on macOS, you may need to install LLVM (see [this post](https://stackoverflow.com/questions/41292059/compiling-cython-with-openmp-support-on-osx) for details).

See [this talk](https://www.youtube.com/watch?v=nCtM4Xg7e4k) at Haystack EU 2019 for more information (the slides are available [here](https://github.com/zalando/public-presentations/blob/master/files/2019-10-28-haystack-learning-to-rank-with-multiple-objectives.pdf)).