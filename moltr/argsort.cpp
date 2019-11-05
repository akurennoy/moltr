#include <algorithm>

#include "argsort.h"



struct IndexedElement {
    int index;
    double value;
};


bool cmp(const IndexedElement & a, const IndexedElement & b){
    return a.value > b.value;
};


void argsort(double* data, int* indices, int n){
	IndexedElement* order_struct = new IndexedElement[n];

	for (int i = 0; i < n; i++){
		order_struct[i].index = i;
		order_struct[i].value = data[i];
	}

	std::stable_sort(&order_struct[0], (&order_struct[0]) + n, &cmp);

	for (int i = 0; i < n; i++){
		indices[i] = order_struct[i].index;
	}

	delete[] order_struct;
};