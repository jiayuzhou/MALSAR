#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h"
#include "epph.h" /* This is the head file that contains the implementation of the used functions*/


/*
-------------------------- Function eplb -----------------------------

 Euclidean Projection onto l1 Ball (eplb)
 
        min  1/2 ||x- v||_2^2
        s.t. ||x||_1 <= z
 
 which is converted to the following zero finding problem
 
        f(lambda)= sum( max( |v|-lambda,0) )-z=0

 For detail, please refer to our paper:

 Jun Liu and Jieping Ye. Efficient Euclidean Projections in Linear Time,
 ICML 2009.  
 
 Usage (in matlab):
 [x, lambda, iter_step]=eplb(v, n, z, lambda0);
 

-------------------------- Function eplb -----------------------------
 */


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double* v=            mxGetPr(prhs[0]);
    int     n=       (int)mxGetScalar(prhs[1]);
    double  z=            mxGetScalar(prhs[2]);
    double  lambda0=      mxGetScalar(prhs[3]);
    
    double *x, *lambda;
    double *iter_step;
    int steps;
    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    
    x=mxGetPr(plhs[0]);
    lambda=mxGetPr(plhs[1]);
    iter_step=mxGetPr(plhs[2]);
    
    eplb(x, lambda, &steps, v, n, z, lambda0);
    *iter_step=steps;    
}

