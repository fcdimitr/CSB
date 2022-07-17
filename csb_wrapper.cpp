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

  void gespmv_double_int64( BiCsb<double,int64_t> * bicsb, double * const x, double * const y ){

    // prepare template type for CSB routine
    typedef PTSR<double,double> PTDD;

    bicsb_gespmv<PTDD>( *bicsb, x, y );

  }


  void deallocate_double_int64( BiCsb<double,int64_t> * bicsb ){
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
