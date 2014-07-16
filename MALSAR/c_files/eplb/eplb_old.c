#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h"

#define delta 1e-12

/*
 Euclidean Projection onto l1 Ball (eplb)
 
        min  1/2 ||x- y||_2^2
        s.t. ||x||_1 <= z
 
which is converted to the following zero finding problem
 
        f(lambda)= sum( max( |y|-lambda,0) )-z=0
 
 Usage:
 [x, lambda, iter_step]=eplb(y, n, z, lambda0);
 
 */

void eplb_new(double * x, double *root, int * steps, double * v,
int n, double z, double lambda0)
{
    
    int i, j, flag=0;
    int rho_1, rho_2, rho, rho_T, rho_S;
    int V_i_b, V_i_e, V_i;
    double lambda_1, lambda_2, lambda_T, lambda_S, lambda;
    double s_1, s_2, s, s_T, s_S, v_max, temp;
    double f_lambda_1, f_lambda_2, f_lambda, f_lambda_T, f_lambda_S;
    int iter_step=0;
        
    /* find the maximal absolute value in v
     * and copy the (absolute) values from v to x
     */

	if (z< 0){
		printf("\n z should be nonnegative!");
		exit(-1);
	}
           
    V_i=0;    
    if (v[0] !=0){
        rho_1=1;
        s_1=x[V_i]=v_max=fabs(v[0]);
        V_i++;
    }
    else{
        rho_1=0;
        s_1=v_max=0;
    }    
    
    for (i=1;i<n; i++){
        if (v[i]!=0){
            x[V_i]=fabs(v[i]); s_1+= x[V_i]; rho_1++; 
            
            if (x[V_i] > v_max)
                v_max=x[V_i];
            V_i++;
        }
    }
    
    /* If ||v||_1 <= z, then v is the solution  */
    if (s_1 <= z){
        flag=1;        lambda=0;
        for(i=0;i<n;i++){
            x[i]=v[i];
        }
        *root=lambda;
        *steps=iter_step;
        return;
    }
    
    lambda_1=0; lambda_2=v_max;
    f_lambda_1=s_1 -z;
    //f_lambda_1=s_1-rho_1* lambda_1 -z;
    rho_2=0; s_2=0; f_lambda_2=-z; 
    V_i_b=0; V_i_e=V_i-1;
    
    lambda=lambda0; 
    if ( (lambda<lambda_2) && (lambda> lambda_1) ){ 
    /*-------------------------------------------------------------------
                  Initialization with the root
     *-------------------------------------------------------------------
     */
           
        i=V_i_b; j=V_i_e; rho=0; s=0;
        while (i <= j){            
            while( (i <= V_i_e) && (x[i] <= lambda) ){
                i++;
            }
            while( (j>=V_i_b) && (x[j] > lambda) ){
                s+=x[j];                
                j--;
            }
            if (i<j){
                s+=x[i];
                
                temp=x[i];  x[i]=x[j];  x[j]=temp;
                i++;  j--;
            }
		}
        
        rho=V_i_e-j;  rho+=rho_2;  s+=s_2;        
		f_lambda=s-rho*lambda-z;
        
        if ( fabs(f_lambda)< delta ){
            flag=1;
		}
		
		if (f_lambda <0){
			lambda_2=lambda; s_2=s;	rho_2=rho; f_lambda_2=f_lambda;

			V_i_e=j;  V_i=V_i_e-V_i_b+1;
		}
		else{
			lambda_1=lambda; rho_1=rho;	s_1=s; f_lambda_1=f_lambda;

			V_i_b=i; V_i=V_i_e-V_i_b+1;
		}

		if (V_i==0){
			//printf("\n rho=%d, rho_1=%d, rho_2=%d",rho, rho_1, rho_2);

            //printf("\n V_i=%d",V_i);
            
			lambda=(s - z)/ rho;
			flag=1;
		}       
     /*-------------------------------------------------------------------
                          End of initialization
      *--------------------------------------------------------------------
      */       
        
    }/* end of if(!flag) */
    
    while (!flag){
        iter_step++;
        
        /* compute lambda_T  */
        lambda_T=lambda_1 + f_lambda_1 /rho_1;
        if(rho_2 !=0){
            if (lambda_2 + f_lambda_2 /rho_2 >	lambda_T)
                lambda_T=lambda_2 + f_lambda_2 /rho_2;
        }
        
        /* compute lambda_S */
        lambda_S=lambda_2 - f_lambda_2 *(lambda_2-lambda_1)/(f_lambda_2-f_lambda_1);
        
        if (fabs(lambda_T-lambda_S) <= delta){
            lambda=lambda_T; flag=1;
            break;
        }
        
        /* set lambda as the middle point of lambda_T and lambda_S */
        lambda=(lambda_T+lambda_S)/2;
        
        s_T=s_S=s=0;
        rho_T=rho_S=rho=0;
        i=V_i_b; j=V_i_e;
        while (i <= j){            
            while( (i <= V_i_e) && (x[i] <= lambda) ){
                if (x[i]> lambda_T){
                    s_T+=x[i]; rho_T++;
                }
                i++;
            }
            while( (j>=V_i_b) && (x[j] > lambda) ){
                if (x[j] > lambda_S){
                    s_S+=x[j]; rho_S++;
                }
                else{
                    s+=x[j];  rho++;
                }
                j--;
            }
            if (i<j){
                if (x[i] > lambda_S){
                    s_S+=x[i]; rho_S++;
                }
                else{
                    s+=x[i]; rho++;
                }
                
                if (x[j]> lambda_T){
                    s_T+=x[j]; rho_T++;
                }
                
                temp=x[i]; x[i]=x[j];  x[j]=temp;
                i++; j--;
            }
		}
        
        s_S+=s_2; rho_S+=rho_2;
        s+=s_S; rho+=rho_S;
        s_T+=s; rho_T+=rho;
        f_lambda_S=s_S-rho_S*lambda_S-z;
        f_lambda=s-rho*lambda-z;
        f_lambda_T=s_T-rho_T*lambda_T-z;
        
        //printf("\n %d & %d  & %5.6f & %5.6f & %5.6f & %5.6f & %5.6f \\\\ \n \\hline ", iter_step, V_i, lambda_1, lambda_T, lambda, lambda_S, lambda_2);
                
        if ( fabs(f_lambda)< delta ){
            //printf("\n lambda");
            flag=1;
            break;
        }
        if ( fabs(f_lambda_S)< delta ){
           // printf("\n lambda_S");
            lambda=lambda_S; flag=1;
            break;
        }
        if ( fabs(f_lambda_T)< delta ){
           // printf("\n lambda_T");
            lambda=lambda_T; flag=1;
            break;
        }        
        
        /*
        printf("\n\n f_lambda_1=%5.6f, f_lambda_2=%5.6f, f_lambda=%5.6f",f_lambda_1,f_lambda_2, f_lambda);
        printf("\n lambda_1=%5.6f, lambda_2=%5.6f, lambda=%5.6f",lambda_1, lambda_2, lambda);
        printf("\n rho_1=%d, rho_2=%d, rho=%d ",rho_1, rho_2, rho);
         */
        
        if (f_lambda <0){
            lambda_2=lambda;  s_2=s;  rho_2=rho;
            f_lambda_2=f_lambda;            
            
            lambda_1=lambda_T; s_1=s_T; rho_1=rho_T;
            f_lambda_1=f_lambda_T;
            
            V_i_e=j;  i=V_i_b;
            while (i <= j){
                while( (i <= V_i_e) && (x[i] <= lambda_T) ){
                    i++;
                }
                while( (j>=V_i_b) && (x[j] > lambda_T) ){
                    j--;
                }
                if (i<j){                    
                    x[j]=x[i];
                    i++;   j--;
                }
            }            
            V_i_b=i; V_i=V_i_e-V_i_b+1;
        }
        else{
            lambda_1=lambda;  s_1=s; rho_1=rho;
            f_lambda_1=f_lambda;
            
            lambda_2=lambda_S; s_2=s_S; rho_2=rho_S;
            f_lambda_2=f_lambda_S;
            
            V_i_b=i;  j=V_i_e;
            while (i <= j){
                while( (i <= V_i_e) && (x[i] <= lambda_S) ){
                    i++;
                }
                while( (j>=V_i_b) && (x[j] > lambda_S) ){
                    j--;
                }
                if (i<j){
                    x[i]=x[j];
                    i++;   j--;
                }
            }
            V_i_e=j; V_i=V_i_e-V_i_b+1;
        }
        
        if (V_i==0){
            lambda=(s - z)/ rho; flag=1;
            //printf("\n V_i=0, lambda=%5.6f",lambda);
            break;
        }
    }/* end of while */
    
    
    for(i=0;i<n;i++){        
        if (v[i] > lambda)
            x[i]=v[i]-lambda;
        else
            if (v[i]< -lambda)
                x[i]=v[i]+lambda;
            else
                x[i]=0;
    }
    *root=lambda;
    *steps=iter_step;
}


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double* v=            mxGetPr(prhs[0]);
    int     n=            mxGetScalar(prhs[1]);
    double  z=            mxGetScalar(prhs[2]);
    double  lambda0=      mxGetScalar(prhs[3]);
    
    double *x, *lambda;
    int *iter_step;
    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[2] = mxCreateNumericMatrix(1,1, mxINT32_CLASS, 0);
    
    x=mxGetPr(plhs[0]);
    lambda=mxGetPr(plhs[1]);
    iter_step=mxGetPr(plhs[2]);
    
    eplb_new(x, lambda, iter_step, v, n, z, lambda0);
}

