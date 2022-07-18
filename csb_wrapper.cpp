/*!
  \file   csb_wrapper.cpp
  \brief  Wrapper for CSB object and routines.

  \author Dimitris Floros
  \date   2019-07-12
*/

// --------------------------------------------------
// Include headers

#define NOMINMAX

#define RHSDIM 1
#define ALIGN 32

#include <iostream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>

#include "timer.gettimeofday.c"
#include "cilk_util.h"
#include "utility.h"

#include "cilk/cilk.h"
#include "cilk/cilk_api.h"

#include "triple.h"
#include "csc.h"
#include "bicsb.h"
#include "bmcsb.h"
#include "spvec.h"
#include "Semirings.h"
#include "aligned.h"


template <typename NT, int DIM>
void fillzero (vector< array<NT,DIM> > & vecofarr)
{
  for(auto itr = vecofarr.begin(); itr != vecofarr.end(); ++itr)
  {
    itr->fill(static_cast<NT> (0));
  }
}

template <typename NT, int DIM>
void fillrandom (vector< array<NT,DIM>> & vecofarr)
{
	for(auto itr = vecofarr.begin(); itr != vecofarr.end(); ++itr)
	{
	#if (__GNUC__ == 4 && (__GNUC_MINOR__ < 7) )
		RandGen G;
		for(auto refarr = itr->begin(); refarr != itr->end(); ++refarr)
		{
			*refarr = G.RandReal();
		}
	#else
		std::uniform_real_distribution<NT> distribution(0.0f, 1.0f); //Values between 0 and 1
		std::mt19937 engine; // Mersenne twister MT19937
		auto generator = std::bind(distribution, engine);
		std::generate_n(itr->begin(), DIM, generator);
	#endif
	}
}


template <typename NT, int DIM>
void copy_to_vector (vector< array<NT,DIM> > & vecofarr, NT * vv, int n)
{
  int i = 0 ;
  for(auto itr = vecofarr.begin(); itr != vecofarr.end(); ++itr)
  {
    int d = 0;
    for(auto refarr = itr->begin(); refarr != itr->end(); ++refarr)
    {
      *refarr = vv[i + n * d];
      d++;
    }
    i++;
  }
}

template <typename NT, int DIM>
void copy_from_vector (vector< array<NT,DIM> > & vecofarr, NT * vv, int n)
{
  int i = 0 ;
  for(auto itr = vecofarr.begin(); itr != vecofarr.end(); ++itr)
  {
    int d = 0;
    for(auto refarr = itr->begin(); refarr != itr->end(); ++refarr)
    {
      vv[i + n * d] = *refarr;
      d++;
    }
    i++;
  }
}

/*
 * Although this is a template, the result is always a CSB object.
 * This is a hack to get arount the CSB library.
 */
template <class NT, class IT>
BiCsb<NT,IT> * prepareCSB( NT *values, IT *rows, IT *cols,
                IT nzmax, IT m, IT n,
                int workers, int forcelogbeta ){

  // generate CSC object (CSB definitions)
  Csc<NT, IT> * csc;
  csc = new Csc<NT, IT>();

  csc->SetPointers( cols, rows, values, nzmax, m, n, 0 );

  workers = __cilkrts_get_nworkers();
  
  BiCsb<NT,IT> *bicsb = new BiCsb<NT, IT>(*csc, workers, forcelogbeta);

  // clean CSB-type CSC object
  operator delete(csc);

  return bicsb;
}


template <class NT, class IT>
void deallocate( BiCsb<NT,IT> * bicsb ){

  // generate CSC object (CSB definitions)
  delete bicsb;
}


template <class NT, class IT, int DIM>
void gespmm( BiCsb<NT,IT> * bicsb,
             NT * const x, NT * const y,
             int nlhs, int nrhs ){

  // prepare template type for CSB routine
  typedef PTSRArray<double, double, DIM> PTARR;
  typedef array<NT, DIM> PACKED;

  vector< PACKED > y_vec(nlhs);
	vector< PACKED > x_vec(nrhs);

  fillzero<NT, DIM>(y_vec);
  copy_to_vector<NT, DIM>(x_vec, x, nrhs);

  bicsb_gespmv<PTARR>( *bicsb, &(x_vec[0]), &(y_vec[0]));

  copy_from_vector<NT, DIM>(y_vec, y, nlhs);


}


template <class NT, class IT, int DIM>
void gespmmt( BiCsb<NT,IT> * bicsb,
              NT * const x, NT * const y,
              int nlhs, int nrhs ){

  // prepare template type for CSB routine
  typedef PTSRArray<double, double, DIM> PTARR;
  typedef array<NT, DIM> PACKED;

  vector< PACKED > y_vec(nlhs);
	vector< PACKED > x_vec(nrhs);

  fillzero<NT, DIM>(y_vec);
  copy_to_vector<NT, DIM>(x_vec, x, nrhs);

  bicsb_gespmvt<PTARR>( *bicsb, &(x_vec[0]), &(y_vec[0]));

  copy_from_vector<NT, DIM>(y_vec, y, nlhs);


}


// ***** EXPLICIT INSTATIATION
extern "C" {

  int getWorkers(){
    return __cilkrts_get_nworkers();
  }

  void setWorkers(unsigned int nWorkers){
    char strw[10];
    sprintf(strw, "%d", nWorkers);
    __cilkrts_end_cilk();
    __cilkrts_set_param("nworkers", strw);
  }


  BiCsb<double, uint32_t> * prepareCSB_double_uint32
  (double *vals, uint32_t *rows, uint32_t *cols,
   uint32_t nzmax, uint32_t m, uint32_t n,
   int workers, int forcelogbeta ){

    return prepareCSB<double, uint32_t>(vals, rows, cols, nzmax, m, n, workers, forcelogbeta);

  }

  BiCsb<double, __int64> * prepareCSB_double_int64
  (double *vals, __int64 *rows, __int64 *cols,
   __int64 nzmax, __int64 m, __int64 n,
   int workers, int forcelogbeta ){

    return prepareCSB<double, __int64>(vals, rows, cols, nzmax, m, n, workers, forcelogbeta);

  }

  void gespmv_double_uint32( BiCsb<double,uint32_t> * bicsb, double * const x, double * const y ){

    // prepare template type for CSB routine
    typedef PTSR<double,double> PTDD;

    bicsb_gespmv<PTDD>( *bicsb, x, y );

  }

  void gespmv_double_int64( BiCsb<double,__int64> * bicsb, double * const x, double * const y ){

    // prepare template type for CSB routine
    typedef PTSR<double,double> PTDD;

    bicsb_gespmv<PTDD>( *bicsb, x, y );

  }

  void gespmvt_double_int64( BiCsb<double,__int64> * bicsb, double * const x, double * const y ){

    // prepare template type for CSB routine
    typedef PTSR<double,double> PTDD;

    bicsb_gespmvt<PTDD>( *bicsb, x, y );

  }

  void gespmvt_double_uint32( BiCsb<double,uint32_t> * bicsb, double * const x, double * const y ){

    // prepare template type for CSB routine
    typedef PTSR<double,double> PTDD;

    bicsb_gespmvt<PTDD>( *bicsb, x, y );

  }

#define DECLARE_GESPMM(DIMS)                                            \
  void gespmm_double_int64_ ##DIMS## _rhs ( BiCsb<double,__int64> * bicsb, \
                                     double * const x, double * const y, \
                                     int nlhs, int nrhs ){              \
                                                                        \
    gespmm<double, __int64, DIMS>( bicsb,                               \
                                   x, y,                                \
                                   nlhs, nrhs );                        \
                                                                        \
  }                                                                     \
                                                                        \
  void gespmmt_double_int64_ ##DIMS## _rhs ( BiCsb<double,__int64> * bicsb, \
                                     double * const x, double * const y, \
                                     int nlhs, int nrhs ){              \
                                                                        \
    gespmmt<double, __int64, DIMS>( bicsb,                               \
                                    x, y,                               \
                                    nlhs, nrhs );                       \
                                                                        \
  }                                                                     \
  void gespmm_double_uint32_ ##DIMS## _rhs ( BiCsb<double,uint32_t> * bicsb, \
                                     double * const x, double * const y, \
                                     int nlhs, int nrhs ){              \
                                                                        \
    gespmm<double, uint32_t, DIMS>( bicsb,                               \
                                   x, y,                                \
                                   nlhs, nrhs );                        \
                                                                        \
  }                                                                     \
                                                                        \
  void gespmmt_double_uint32_ ##DIMS## _rhs ( BiCsb<double,uint32_t> * bicsb, \
                                     double * const x, double * const y, \
                                     int nlhs, int nrhs ){              \
                                                                        \
    gespmmt<double, uint32_t, DIMS>( bicsb,                               \
                                    x, y,                               \
                                    nlhs, nrhs );                       \
                                                                        \
  }

  DECLARE_GESPMM(1)
  DECLARE_GESPMM(2)
  DECLARE_GESPMM(3)
  DECLARE_GESPMM(4)
  DECLARE_GESPMM(5)
  DECLARE_GESPMM(6)
  DECLARE_GESPMM(7)
  DECLARE_GESPMM(8)
  DECLARE_GESPMM(9)
  DECLARE_GESPMM(10)
  DECLARE_GESPMM(11)
  DECLARE_GESPMM(12)
  DECLARE_GESPMM(13)
  DECLARE_GESPMM(14)
  DECLARE_GESPMM(15)
  DECLARE_GESPMM(16)
  DECLARE_GESPMM(17)
  DECLARE_GESPMM(18)
  DECLARE_GESPMM(19)
  DECLARE_GESPMM(20)
  DECLARE_GESPMM(21)
  DECLARE_GESPMM(22)
  DECLARE_GESPMM(23)
  DECLARE_GESPMM(24)
  DECLARE_GESPMM(25)
  DECLARE_GESPMM(26)
  DECLARE_GESPMM(27)
  DECLARE_GESPMM(28)
  DECLARE_GESPMM(29)
  DECLARE_GESPMM(30)
  DECLARE_GESPMM(31)
  DECLARE_GESPMM(32)



  void deallocate_double_int64( BiCsb<double,__int64> * bicsb ){
    deallocate( bicsb );
  }

  void deallocate_double_uint32( BiCsb<double,uint32_t> * bicsb ){
    deallocate( bicsb );
  }


}

/**------------------------------------------------------------
*
* AUTHORS
*
*   Dimitris Floros                         fcdimitr@auth.gr
*
* VERSION
* 
*   1.0 - July 13, 2018
*
* CHANGELOG
*
*   1.0 (Jul 13, 2018) - Dimitris
*       * all interaction types in one file
*       
* ----------------------------------------------------------*/
