/* ---------------------------------------------------------- */
/* mexFunction: segL2                                         */
/*                                                            */
/* compute L2 norm for each provided segment                  */
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
        mexErrMsgTxt ("Usage: out = segL2 (vect, accIdx)") ;
    
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
    pargout [0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    v = mxGetPr( pargout [0] );
    v[0] = 0.0;
    
    /* compute segment sum. */
    for (ii = 0; ii< segNum-1; ii++){
        t = 0.0;
        for (jj = accIdx[ii]; jj < accIdx[ii+1]; jj++){
            t += vect[jj] * vect[jj];
        }
        v[0]+= sqrt(t);
    }
    return;
}