#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h"

#define tol 1e-8

/*
-------------------------- Function prf_lb -----------------------------

 Euclidean Projection onto l1 Ball (prf_lb)
 
        min  1/2 ||x- c||_2^2
        s.t. ||x||_1 <= t

 Usage (in matlab):
 [x, theta, iter]=prf_lb(c, m, n, t); n is the dimension of the vector c

-------------------------- Function prf_lb -----------------------------
 */

void prf_lbm(double *x, double *root, double *steps, double *c, int m, int n, double t)
{
	int i, j, gnum, rho = 0;
	double theta = 0;
	double s = 0;
	double iter_step=0; 

	if (t < 0)
	{
		printf("\n t should be nonnegative!");
		return;
	}
	
	for (j=0;j<m;j++)
	{
		gnum = 0; theta = 0; s = 0;
		iter_step = 0; rho = 0;
		for(i=0;i<n;i++)
		{
			//if (c[i]!=0)
			//{
				x[gnum*m+j] = fabs(c[i*m+j]);
				s += x[gnum*m+j];
				gnum++;
			//}
		}

		/* If ||c||_1 <= t, then c is the solution  */
		if (s <= t)
		{
			theta=0;
			for(i=0;i<n;i++)
			{
				x[i*m+j]=c[i*m+j];
			}
			root[j]=theta;
			steps[j]=iter_step;
			continue;
		}

		/*while loops*/
		while (fabs(s - t - gnum*theta) > tol)
		{
			iter_step++;
			theta = (s-t)/gnum;
			s=0; rho = 0;
			for (i=0;i<gnum;i++)
			{
				if (x[i*m+j] >= theta)
				{
					x[rho*m+j] = x[i*m+j];
					s+=x[i*m+j]; rho++;
				}
			}
			gnum = rho;

     

		}/*end of while*/

		/*projection result*/
		for(i=0;i<n;i++)
		{        
			/*if (c[i] > theta)
				x[i]=c[i]-theta;
			else
				if (c[i]< -theta)
					x[i]=c[i]+theta;
				else
					x[i]=0;*/
			x[i*m+j] = (c[i*m+j] > theta)?(c[i*m+j]-theta):((c[i*m+j]< -theta)?(c[i*m+j]+theta):0);
		}
		root[j]=theta;
		steps[j]=iter_step;
		//printf("step=%d\n",*steps);
	}
	return;
}


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double* c=            mxGetPr(prhs[0]);
	int     m=       (int)mxGetScalar(prhs[1]);
    int     n=       (int)mxGetScalar(prhs[2]);
    double  t=            mxGetScalar(prhs[3]);
    
    double *x, *theta;
	double *iter_step;


    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix(m,n,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(m,1,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(m,1,mxREAL);
    
    x=mxGetPr(plhs[0]);
    theta=mxGetPr(plhs[1]);
	iter_step= mxGetPr(plhs[2]);
    
    prf_lbm(x, theta, iter_step, c, m, n, t);
    //printf("step=%f\n",*iter_step);
}


