#ifndef __SWIG_CPP
#define __SWIG_CPP

#include <stdlib.h>
#include "swig_mod.h"

#include <iostream>
#include <cmath>
using namespace std;

void cholesky_swig(int mm1, int mn1, double* mat1,
              double** outmat, int* mm, int* mn)
{
    double* arr = NULL;
    arr = (double*)calloc(mn1 * mm1, sizeof(double));
    
    for(int i = 0; i < mm1; ++i){
        for(int j = 0; j < i+1; ++j){
            double s = 0;
            for(int k=0; k < j; ++k)
                s += arr[i*mn1+k]*arr[j*mn1+k];
            
            if(i==j){
                arr[i*mn1+j] = sqrt(mat1[i*mn1+i] - s);
            } else {
                arr[i*mn1+j] = (mat1[i*mn1+j] - s)/arr[j*mn1+j];
            }
        }

    }
    *mm = mm1;
    *mn = mn1;
    *outmat = arr;
}
#endif
