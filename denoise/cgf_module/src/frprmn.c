/* 
 * frprmn.c:	modified frprmn for efficient line minimization
 */
#include <math.h>
#include "nrutil.h"

#define EPS 1.0e-10
#define FREEALL free_vector(xi,1,n);free_vector(h,1,n);free_vector(g,1,n);

// int ITMAX;

void frprmn(p,n,ftol,iter,fret,init_f1dim,f1dim,dfunc, itmax)
float (*init_f1dim)(),(*f1dim)(),*fret,ftol,p[];
int *iter,n, itmax;
void (*dfunc)();
{
	void linmin();
	int j,its;
	float gg,gam,fp,dgg;
	float *g,*h,*xi;

	g=vector(1,n);
	h=vector(1,n);
	xi=vector(1,n);
    
    // Compute the initial gradient
	(*dfunc)(p,xi);
    
    // Set the initial search direction
	for (j=1;j<=n;j++) {
		g[j] = -xi[j];
		xi[j]=h[j]=g[j];
	}
	for (its=1;its<=itmax;its++) {
		*iter=its;
		fp=(*init_f1dim)(p,xi);
        
        // Compute the optimal step length alpha_{k} by performing a line minimization along the direction d_{k}. 
        // This involves minimizing the 1-dimensional function h(alpha)=f(x_{k} + alpha * d_{k})).
		linmin(p,xi,n,fret,f1dim);
        
        // Calculate the new gradient g_{k+1}=grad(f(x_{k+1})). If the norm of g_{k+1} is below a specified tolerance, then stop (a minimum has been found).
		if (2.0*fabs(*fret-fp) <= ftol*(fabs(*fret)+fabs(fp)+EPS)) {
			FREEALL
			return;
		}

        (*dfunc)(p,xi);

        // Compute New Conjugate Direction
		dgg=gg=0.0;
		for (j=1;j<=n;j++) {
			gg += g[j]*g[j];
			dgg += (xi[j]+g[j])*xi[j];
		}
		if (gg == 0.0) {
			FREEALL
			return;
		}
		gam=dgg/gg;
		for (j=1;j<=n;j++) {
			g[j] = -xi[j];
			xi[j]=h[j]=g[j]+gam*h[j];
		}
	}
/*	nrerror("Too many iterations in frprmn"); */
	FREEALL
	return;
}

#undef EPS
#undef FREEALL
/* (C) Copr. 1986-92 Numerical Recipes Software 6=Mn.Y". */
