#! /bin/bash

cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.1/g" > aiai_seg_ul_sweep_0.1.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.2/g" > aiai_seg_ul_sweep_0.2.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.3/g" > aiai_seg_ul_sweep_0.3.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.4/g" > aiai_seg_ul_sweep_0.4.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.5/g" > aiai_seg_ul_sweep_0.5.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.6/g" > aiai_seg_ul_sweep_0.6.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.7/g" > aiai_seg_ul_sweep_0.7.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.8/g" > aiai_seg_ul_sweep_0.8.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.9/g" > aiai_seg_ul_sweep_0.9.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.95/g" > aiai_seg_ul_sweep_0.95.sbatch
cat aiai_seg_ul_sweep.sbatch.template | sed "s/UL_FRAC/0.99/g" > aiai_seg_ul_sweep_0.99.sbatch

ls *.sbatch | xargs -n 1 sbatch
