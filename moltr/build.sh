#!/bin/sh

export CC=/usr/local/opt/llvm/bin/clang++

if [ -f "$CC" ]; then
	"$CC" -c argsort.cpp
	ar rcs libargsort.a argsort.o
	rm argsort.o
	python setup.py build_ext --inplace
	rm lambdaobj.c libargsort.a
	rm -rf build
else
	echo "LLVM is required (please run 'brew install llvm')."
fi
