/*
 * cgf.c:	conj. grad. routine for finding optimal s - fast!
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <math.h>

#define sgn(x) (x>0 ? 1 : (x<0 ? -1 : 0))

// frprmn	minimize in N-dimensions by conjugate gradient
// mnbrak	bracket the minimum of a function
// linmin	minimum of a function along a ray in N-dimensions
// brent 	find minimum of a function by Brent’s method

// To "bracket the minimum of a function" means finding three points, a<b<c, such that the function's value 
// at the middle point is lower than at the ends: f(b)<f(a) and f(b)<f(c), guaranteeing a local minimum exists 
// in the interval [a,c]. This is a common first step in numerical optimization, often using methods
// like ??? or ??? to find this initial interval, then iteratively narrowing it down to pinpoint the minimum's location. 

// Brent's method is a robust and efficient hybrid algorithm used to find the minimum of a function of a single variable 
// within a given interval. It combines the guaranteed convergence of the golden-section search method with the faster 
// convergence of successive parabolic interpolation (SPI). 

extern void cgf(double *Sout, double *nits, double *nf, double *ng,
	       double *Sin, double *X, int npats, 
	       double tol, int maxiter, int numflag);

/* Input & Output Arguments */

#define	A_IN		prhs[0]		/* basis matrix */
#define	X_IN		prhs[1]		/* data vectors */
#define	S_IN		prhs[2]		/* initial guess for S */
#define	LAMBDA_IN	prhs[3]		/* precision */
#define BETA_IN		prhs[4]		/* prior steepness */
#define SIGMA_IN        prhs[5]         /* scaling parameter for prior */
#define	TOL_IN		prhs[6]		/* tolerance */
#define MAXITER_IN      prhs[7]		/* maximum iterations for dfrpmin */
#define	OUTFLAG_IN	prhs[8]		/* output flag */
#define	NUMFLAG_IN	prhs[9]		/* pattern number output flag */

#define	S_OUT           plhs[0]		/* basis coeffs for each data vector */
#define NITS_OUT        plhs[1]         /* total iterations done by cg */
#define NF_OUT          plhs[2]         /* total P(s|x,A) calcs */
#define NG_OUT          plhs[3]         /* total d/ds P(s|x,A) calcs */

/* Define indexing macros for matricies */

/* L = dimension of input vectors
 * M = number of basis functions
 */

// Column-major matrices layout, Fortran-style
#define A_(i,j)		A[(i) + (j)*L]		/* A is L x M */
#define X_(i,n)		X[(i) + (n)*L]		/* X is L x npats */

#define Sout_(i,n)	Sout[(i) + (n)*M]	/* S is M x npats */
#define Sin_(i,n)	Sin[(i) + (n)*M]	/* S is M x npats */

#define AtA_(i,j)	AtA[(i) + (j)*M]	/* AtA is M x M */

/* Globals for using with frprmin */

static double *A;		/* basis matrix */
static int L;			/* data dimension */
static int M;			/* number of basis vectors */
static double lambda;		/* 1/noise_var */
static double beta;		/* prior steepness */
static double sigma;		/* prior scaling */
static double k1,k2,k3;		/* precomputed constants for f1dim */

static double *x;		/* current data vector being fitted */
static double *s0;		/* init coefficient vector (1:M) */
static double *d;		/* search dir. coefficient vector (1:M) */
static int outflag;		/* print search progress */

static double *AtA;		/* Only compute A'*A once (1:M,1:M) */
static double *Atx;		/* A*x (1:M) */

static int fcount, gcount;

static void init_global_arrays() 
{
  int 		i,j,k;
  double	*Ai, *Aj, sum;

  x      = (double *)malloc(L*sizeof(double));
  s0     = (double *)malloc(M*sizeof(double));
  d      = (double *)malloc(M*sizeof(double));
  AtA    = (double *)malloc(M*M*sizeof(double));
  Atx    = (double *)malloc(M*sizeof(double));

  /* Calc  A'*A */
  for (i = 0; i < M; i++) {
    Ai=A+i*L;
    for (j = 0; j < M; j++) {
      Aj=A+j*L;
      sum=0.0;
      for (k = 0; k < L; k++) {
	sum += Ai[k]*Aj[k];
      }
      AtA_(i,j) = sum;
    }
  }
}

static void free_global_arrays() {

  free((double *)x);
  free((double *)s0);
  free((double *)d);
  free((double *)AtA);
  free((double *)Atx);
}

// Conj. gradient method предполагает минимизацию 1-размерной функции h(alpha). Эта
// функция должна каким-то образом сворачивать многомерную нашу функцию. Для этого и существуют init_f1dim/f1dim

float init_f1dim(s1,d1)
float *s1,*d1;
{
    // fprintf(stdout, "init_f1dim: init_f1dim\n");
    // fflush(stdout);
    
  register int		i,j;
  register double	As,Ag,sum;
  register float	fval;
  extern double		sparse();

  for (i=0; i < M; i++) {
    s0[i]=s1[i+1];
    d[i]=d1[i+1];

      // fprintf(stdout, "init_f1dim: s0[%d]=%f, d[%d]=%f\n", i, s0[i], i, d[i]);
      // fflush(stdout);
      
  }
  k1=k2=k3=0;
  for (i=0; i<L; i++) {
    As=Ag=0;
    for (j=0; j<M; j++) {
      As += A_(i,j) * s0[j];
      Ag += A_(i,j) * d[j];
    }
      // Похоже на коэффициенты производных (разложение в ряд a-la Тейлор???)
    k1 += As*(As - 2*x[i]);
    k2 += Ag*(As - x[i]);
    k3 += Ag*Ag;
  }
  k1 *= 0.5*lambda;
  k2 *= lambda;
  k3 *= 0.5*lambda;

  fval = k1;

  sum=0;
  for (i=0; i<M; i++) // for each dimension
    sum += sparse(s0[i]/sigma);
  fval += beta*sum;

  fcount++;

    // fprintf(stdout, "init_f1dim: fval=%f\n", fval);
    //   fflush(stdout);
    
  return(fval);
}

// Это и есть h(alpha), которая минимизируется в frprmn. Она полагается на состояние (k1, k2, k3, s0, d, ...), вычисленное в init_f1dim

float f1dim(alpha)
float	alpha;
{
    // fprintf(stdout, "f1dim: f1dim(%f)\n", alpha);
    // fflush(stdout);
    
  int 		i;
  double 	sum;
  float		fval;
  extern double	sparse();

  fval = k1+(k2+k3*alpha)*alpha;

  sum=0;
  for (i=0; i<M; i++) { // for each dimension
    sum += sparse((s0[i]+alpha*d[i])/sigma);
  }
  fval += beta*sum;

  fcount++;

  return(fval);
}


/*
 * Gradient evaluation used by conj grad descent
 */
void dfunc(p,grad)
float	*p,*grad;
{
  register int		i,j;
  register double	sum,*cptr,bos=beta/sigma;
  register float	*p1;
  extern double		sparse_prime();

  p1=&p[1];

  for (i=0; i<M; i++) {
    cptr=AtA+i*M;
    sum=0;
    for (j=0; j<M; j++) {
      sum += p1[j] * *cptr++;
    }
    grad[i+1] = lambda*(sum - Atx[i]) + bos*sparse_prime((double)p1[i]/sigma);

    // fprintf(stdout, "dfunc: grad[%d+1]=%f, sum=%f, Atx[%d]=%f\n", i, grad[i+1], sum, i, Atx[i]);
    // fflush(stdout);
  }
  gcount++;
}

// Функция для регуляризации разреженности
double sparse(x)
double	x;
{
  return(log(1.0+x*x));
}

// Производная от функции sparse(x)
double sparse_prime(x)
double	x;
{
  return(2*x/(1.0+x*x));
}

void iter_do()
{
}


#include <nrutil.h>
// extern int	ITMAX;

void cgf(double *Sout, double *nits, double *nf, double *ng, 
	double *Sin, double *X, int npats, 
	double tol, int maxiter, int numflag)
{
    double	sum;
    float 	fret;
    int 		niter,l,m,n;
    float 	*p;

    *nits = *nf = *ng = 0.0;
    int itmax = 10;

    init_global_arrays();
    p=vector(1,M);

    for (n = 0; n < npats; n++) { // for each patch
        if (numflag) {
            fprintf(stdout,"cgf: %d\n",n+1);
            fflush(stdout);
        }

        for (l = 0; l < L; l++) { // for each dim of base func / patch
            x[l] = X_(l,n);
            // fprintf(stdout, "x[%d]=%f\n", l, x[l]);
            // fflush(stdout);
        }

        for (m = 0; m < M; m++) { // for each base func
            /* precompute Atx for this pattern */
            sum = 0.0;
      
            for (l = 0; l < L; l++) {
                sum += A_(l,m) * x[l];
            }
              
            Atx[m] = sum;
            // fprintf(stdout, "cgf: Atx[%d]=%f\n", m, Atx[m]);
            // fflush(stdout);

            /* copy initial guess */
            p[m+1] = Sin_(m,n);

            // fprintf(stdout, "cgf: p[%d+1]=%f\n", m, p[m+1]);
            // fflush(stdout);
        }

        fcount=gcount=0;

        frprmn(p, M, (float)tol, &niter, &fret, init_f1dim, f1dim, dfunc, itmax);

        *nits += (double)niter;
        *nf += (double)fcount;
        *ng += (double)gcount;

        if (outflag) {
            fprintf(stdout,"cgf: fret=%f  niters=%d  fcount=%d  gcount=%d\n", fret, niter, fcount, gcount);
            fflush(stdout);
        }

        /* copy back solution */
        for (m = 0; m < M; m++) {
            // fprintf(stdout, "cgf: Sout_(%d,%d)=%f\n", m, n, p[m+1]);
            // fflush(stdout);
            Sout_(m,n) = p[m+1];
        }
    }

    free_global_arrays();
    free_vector(p,1,M);
}

#ifndef PY_SSIZE_T_CLEAN

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  double *Sout, nits=0, nf=0, ng=0, *Sin;
  double *X, tol;
  int maxiter, npats, numflag, i;

  /* Check for proper number of arguments */

  if (nrhs < 6) {
    mexErrMsgTxt("cgf requires 6 input arguments.");
  } else if (nlhs < 1) {
    mexErrMsgTxt("cgf requires 1 output argument.");
  }

  /* Assign pointers to the various parameters */

  A = mxGetPr(A_IN); // +
  X = mxGetPr(X_IN); // +
  Sin = mxGetPr(S_IN); // +
  lambda = mxGetScalar(LAMBDA_IN); // +
  beta = mxGetScalar(BETA_IN); // +
  sigma = mxGetScalar(SIGMA_IN); // +

  if (nrhs < 7) {
    tol = 0.1; // +
  } else {
    tol = mxGetScalar(TOL_IN); // +
  }

  if (nrhs < 8) {
    maxiter = 100; 
  } else {
    maxiter = (int)mxGetScalar(MAXITER_IN);
  }

  if (nrhs < 9) {
    outflag = 0;
  } else {
    outflag = (int)mxGetScalar(OUTFLAG_IN);
  }

  if (nrhs < 10) {
    numflag = 0;
  } else {
    numflag = (int)mxGetScalar(NUMFLAG_IN);
  }

  L = (int)mxGetM(A_IN); // +
  M = (int)mxGetN(A_IN); // +
  npats = (int)mxGetN(X_IN); // +

  /* Create a matrix for the return argument */

  S_OUT = mxCreateDoubleMatrix(M, npats, mxREAL); // +
  Sout = mxGetPr(S_OUT);

  if (nlhs > 1) {
    NITS_OUT = mxCreateDoubleMatrix(1, 1, mxREAL);
  }
  if (nlhs > 2) {
    NF_OUT = mxCreateDoubleMatrix(1, 1, mxREAL);
  }
  if (nlhs > 3) {
    NG_OUT = mxCreateDoubleMatrix(1, 1, mxREAL);
  }

  /* Do the actual computations in a subroutine */

  cgf(Sout, &nits, &nf, &ng, Sin, X, npats, tol, maxiter, numflag);

  if (nlhs > 1) {
    *(mxGetPr(NITS_OUT)) = nits;
  }
  if (nlhs > 2) {
    *(mxGetPr(NF_OUT)) = nf;
  }
  if (nlhs > 3) {
    *(mxGetPr(NG_OUT)) = ng;
  }
}

#else

static PyObject *
cgf_wrap(PyObject *, PyObject * theArgs) {
    // L = dimension of input vectors
    // M = number of basis functions
    Py_buffer matrixABuf;     // shape: L x M
    Py_buffer matrixXBuf;     // shape: L x npats
    Py_buffer matrixSInitBuf; // shape: M x npats
    Py_buffer matrixSOutBuf;  // shape: M x npats
    int npats;
    double tol = 0;
    int maxIter = 0;
    int numflag = 0;
    outflag = 0;
    
    if(!PyArg_ParseTuple(theArgs, "iiiy*y*y*y*ddddi", &L, &M, &npats, &matrixABuf, &matrixXBuf, &matrixSInitBuf, &matrixSOutBuf, &lambda, &beta, &sigma, &tol, &maxIter))
        return NULL;

    if(matrixABuf.buf && matrixXBuf.buf && matrixSInitBuf.buf && matrixSOutBuf.buf) {
        A = (double *)matrixABuf.buf;
        const double * const X  = (double *)matrixXBuf.buf;
        const double * const Sin = (double *)matrixSInitBuf.buf;
        double * const Sout = (double *)matrixSOutBuf.buf;

        double nits = 0, nf = 0, ng = 0; // unused

        // =====
        // fprintf(stdout, "lambda=%f, beta=%f, sigma=%f\n", lambda, beta, sigma);
        // fflush(stdout);
        
        // fprintf(stdout, "A\n");
        // fflush(stdout);
        // int i, j;

        // for (i=0; i<L; i++) {
        //     for (j=0; j<M; j++) {
        //         fprintf(stdout, "%.2f\t", A_(i,j));
        //         fflush(stdout);
        //     }

        //     fprintf(stdout, "\n");
        //     fflush(stdout);
        // }

        // fprintf(stdout, "X\n");
        // fflush(stdout);

        // for (i=0; i<L; i++) {
        //     for (j=0; j<npats; j++) {
        //         fprintf(stdout, "%.2f\t", X_(i,j));
        //         fflush(stdout);
        //     }

        //     fprintf(stdout, "\n");
        //     fflush(stdout);
        // }

        // fprintf(stdout, "Sin\n");
        // fflush(stdout);

        // for (i=0; i<M; i++) {
        //     for (j=0; j<npats; j++) {
        //         fprintf(stdout, "%.2f\t", Sin_(i,j));
        //         fflush(stdout);
        //     }

        //     fprintf(stdout, "\n");
        //     fflush(stdout);
        // }
        // // =====

        // fprintf(stdout, "tol=%f, maxiter=%d, numflag=%d\n", tol, maxIter, numflag);
        // fflush(stdout);
        cgf(Sout, &nits, &nf, &ng, Sin, X, npats, tol, maxIter, numflag);
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&matrixABuf);
    PyBuffer_Release(&matrixXBuf);
    PyBuffer_Release(&matrixSInitBuf);
    PyBuffer_Release(&matrixSOutBuf);
    return Py_None;
}

static PyObject *
cgf_test(PyObject *, PyObject * theArgs) {
    Py_buffer matrixXBuf;
    int nrows = 0, ncols = 0;
    double d = 0;
    int i = 0, j = 0;
    
    if(!PyArg_ParseTuple(theArgs, "iiy*d", &nrows, &ncols, &matrixXBuf, &d))
        return NULL;

    if(matrixXBuf.buf) {
        fprintf(stdout, "nrows=%d, ncols=%d, d=%.2f\n", nrows, ncols, d);
        fflush(stdout);
        
        const double * const matX = (double *)matrixXBuf.buf;

        for(i = 0; i < nrows; i++) {
            const double * const row = &matX[i * ncols];
            
            for(j = 0; j < ncols; j++) {
                fprintf(stdout, "%.2f\t", row[j]);
                fflush(stdout);
            }

            fprintf(stdout, "\n");
            fflush(stdout);
        }
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&matrixXBuf);
    return Py_None;
}

static PyObject *
cgf_test2(PyObject *, PyObject * theArgs) {
    Py_buffer matrixABuf;
    
    if(!PyArg_ParseTuple(theArgs, "iiy*", &L, &M, &matrixABuf))
        return NULL;

    if(matrixABuf.buf) {
        fprintf(stdout, "L=%d, M=%d\n", L, M);
        fflush(stdout);
        
        A = (double *)matrixABuf.buf;
        int i, j;

        for (i=0; i<L; i++) {
            for (j=0; j<M; j++) {
                fprintf(stdout, "%.2f\t", A_(i,j));
                fflush(stdout);
            }

            fprintf(stdout, "\n");
            fflush(stdout);
        }
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&matrixABuf);
    return Py_None;
}

#endif


#undef A_
#undef X_
#undef Sout_
#undef Sin_
#undef AtA_

// ==========

static PyMethodDef cgf_methods[] = {
    {"cgf", cgf_wrap, METH_VARARGS, ""},
    {"cgf_test", cgf_test, METH_VARARGS, ""},
    {"cgf_test2", cgf_test2, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef cgf_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "cgf",
    .m_doc = "conj. grad. routine for finding optimal s - fast!",
    .m_size = 0,  
    .m_methods = cgf_methods,
};

PyMODINIT_FUNC 
PyInit_cgf(void) {
    return PyModuleDef_Init(&cgf_module);
}

