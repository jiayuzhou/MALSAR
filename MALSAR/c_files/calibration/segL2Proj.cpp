/* ---------------------------------------------------------- */
/* mexFunction: segL2Proj                                     */
/*                                                            */
/* compute L2 norm projection for each provided segment       */
/*                                                            */
/* Jiayu Zhou (jiayu.zhou@asu.edu)                            */
/* ---------------------------------------------------------- */

#include "mex.h"
#include <math.h>

void mexFunction
        (
        int nargout,                  /*number of output variables*/
        mxArray *pargout [ ],         /*pointer of output variables*/
        int nargin,                   /*number of input variables*/
        const mxArray *pargin  [ ]    /*pointer of input variables*/
        ){
    
    if (nargin != 2 || nargout > 1)
        mexErrMsgTxt ("Usage: out = segL2Proj (vect, accIdx)") ;
    
    /* define variables */
    double *vect, *accIdx, *v; /*input variable*/
    double t;
    size_t vectLen, segNum, segDim;
    size_t ii, jj;
    
    /* get inputs from mex */
    vect    = mxGetPr( pargin [0] );
    accIdx  = mxGetPr( pargin [1] );
    
    vectLen = mxGetNumberOfElements( pargin [0] );
    segNum  = mxGetNumberOfElements( pargin [1] );
    
    /* validate input*/
    if ( accIdx[segNum - 1] > vectLen )
        mexErrMsgTxt ("INDEX OUT OF BOUND: the last index is out of bound.");
    
    /* allocate memory for output*/
    pargout [0] = mxCreateDoubleMatrix(vectLen, 1, mxREAL);
    v = mxGetPr( pargout [0] );
    
    for (ii = 0; ii< vectLen; ii++){
        v[ii] = vect[ii];
    }
    
    /* compute segment sum. */
    for (ii = 0; ii< segNum-1; ii++){
        /*compute l2 norm. */
        t = 0.0;
        for (jj = accIdx[ii]; jj < accIdx[ii+1]; jj++){
            t += vect[jj] * vect[jj];
        }
        t = sqrt(t);
        /*projection if l2 norm is larger than 1*/
        if (t> 1){
            for (jj = accIdx[ii]; jj < accIdx[ii+1]; jj++){
                v[jj] = v[jj]/t;
            }
        }
    }
    
    return;
}