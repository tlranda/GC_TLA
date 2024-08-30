/**
 * syr2k.cuh: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef SYR2K_H
# define SYR2K_H

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(NI) && !defined(NJ)
/* Define the possible dataset sizes. Similar scaling to CPU sizes */
#  ifdef MINI_DATASET
#define NI 256
#define NJ 384
#  endif

#  ifdef SMALL_DATASET
#define NI 512
#define NJ 640
#  endif

#  ifdef SM_DATASET
#define NI 1024
#define NJ 1280
#  endif

#  ifdef MEDIUM_DATASET
#define NI 2048
#define NJ 2560
#  endif

#  ifdef ML_DATASET
#define NI 4096
#define NJ 5120
#  endif

#  ifdef LARGE_DATASET /* Default if unspecified. */
#define NI 8192
#define NJ 10240
#  endif

#  ifdef EXTRALARGE_DATASET
#define NI 16384
#define NJ 20480
#  endif
# endif /* !N */

#  ifdef HUGE_DATASET
#define NI 32768
#define NJ 40960
#  endif

# define _PB_NI POLYBENCH_LOOP_BOUND(NI,ni)
# define _PB_NJ POLYBENCH_LOOP_BOUND(NJ,nj)

# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

#endif /* !SYR2K*/

