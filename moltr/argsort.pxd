cdef extern from "argsort.h":
    cdef void argsort(double* data, int* indices, int n) nogil