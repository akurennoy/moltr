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