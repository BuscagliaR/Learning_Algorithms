# Learning Algorithms

## Author and Date
Dr. Robert Buscaglia (rbuscagl@asu.edu)

April 2018

## Description
 
This repository contains supervised learning and ensemble algorithms used in the dissertation "Supervised and Ensemble Learning of Multivariate Functional Data: Applications to Lupus Diagnosis", defended on May 10th, 2018, Arizona State University.

Provided are the functions used to produce  supervised learning results presented within the dissertation.  This relates to Chapters 4, 5, and 7, with code given for how KCV for contemporary and functional classifiers were produced.  Limited specific code is given for obtaining values found within tables or figures.  Requests can be made to the author if specific code of interest is required.

The author hopes to maintain this repository and will provide more code for producing specific results with time.  As for now, only minimal code has been provided for running and obtain results found within the dissertation.  These are given only to provide a template for producing other results or using the algorithms created.

Annotations are often also missing.  Updates to annotations are also of future interest, so that fundamental code can be sharedat the author's new position.

## Supervised_Learning_Algorithms.R

This file contains all supervised learning algorithms.  Algorithms are all created to perform k-fold cross validation.  Some parallelization is present within the functions.  Classifiers include : MLE logistic regression (LR), LASSO, ENET, RIDGE, LDA, QDA, and KNN.  Adaptive varients of ENET and LASSO were coded seperately.

Provided within this set of code are basic naive, equally-weighted, and accuracy-weighted ensemble algorithms.

## Ensemble_Functions.R

Contains R code for ensemble learning.  Naive, equally-weighted, and accuracy-weighted ensemble algorithms are presented, but now with detailed code for running and producing KCV ensemble classifier results found in Chapter 5.

## ESFuNC_Functions_March_2018.R

ESFuNC functions (see additional repository).  Not required but called in the code provided to obtain results.

## lupustherm.txt

SLE Plasma Thermogram averaged scans.  Data was provided by Dr. Nichola Garbett.  The file contains the SLE plasma thermogram output after some data processing: averaging of duplicated results and removal of erroneous scans.

## Per_Pixel_SLE_Full_Analysis_Final.R

Provides the code for obtaining all results shown in Chapter 7, section 3.  Not annotated.  Uses parallelization heavily (6 cores).  Several computationally intensive steps.  Takes approximately 6 - 8 hours to run.

