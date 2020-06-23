%module SwigMod

%{
    #define SWIG_FILE_WITH_INIT
    #include "swig_mod.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2){(double** outmat, int* mm, int* mn)}
%apply (int DIM1, int DIM2, double* IN_ARRAY2){(int mm1, int mn1, double* mat1)}

%include "swig_mod.h"
