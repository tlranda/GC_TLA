clang -S -emit-llvm amg2013.c -o amg.s -I.. -I../utilities -I../struct_mv -I../sstruct_mv -I../IJ_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -I../krylov -O2 -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -DHYPRE_BIGINT -DHYPRE_TIMING

opt -S -polly-canonicalize amg.s > amg.preopt.ll

opt -polly-ast -analyze -q amg.preopt.ll -polly-process-unprofitable

opt -view-scops -disable-output amg.preopt.ll

opt -polly-scops -analyze amg.preopt.ll -polly-process-unprofitable

opt -polly-dependences -analyze amg.preopt.ll -polly-process-unprofitable

opt amg.preopt.ll -polly-import-jscop -polly-ast -analyze -polly-process-unprofitable

opt amg.preopt.ll -polly-import-jscop -polly-import-jscop-postfix=interchanged -polly-ast -analyze -polly-process-unprofitable

opt amg.preopt.ll -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled -polly-ast -analyze -polly-process-unprofitable


opt amg.preopt.ll | opt -O3 > amg.normalopt.ll

llc amg.normalopt.ll -o amg.normalopt.s

 gcc amg.normalopt.s -o amg.normalopt

clang -O3 -march=native -ffast-math -mllvm -polly amg2013.c -I.. -I../utilities -I../struct_mv -I../sstruct_mv -I../IJ_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -I../krylov  -mllvm -debug-only=polly-scops -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -DHYPRE_BIGINT -DHYPRE_TIMING -S -emit-llvm


clang -O3 -march=native -ffast-math -mllvm -polly amg2013.c -I.. -I../utilities -I../struct_mv -I../sstruct_mv -I../IJ_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -I../krylov  -Rpass=polly-scops -Rpass-analysis=polly-scops -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -DHYPRE_BIGINT -DHYPRE_TIMING -c -o amg2013.o

clang -O3 -march=native -ffast-math -mllvm -polly amg2013.c -I.. -I../utilities -I../struct_mv -I../sstruct_mv -I../IJ_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -I../krylov  -Rpass=polly-scops -Rpass-analysis=polly-scops -DTIMER_USE_MPI -DHYPRE_USING_OPENMP -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -DHYPRE_BIGINT -DHYPRE_TIMING -c -o amg2013.o -mllvm -debug-only=polly-ast > amg.txt 2>&1

