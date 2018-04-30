### Per-Pixel Analysis
### Dr. Robert Buscaglia
### March 2018
### Evaluate classifiers at each predictor, producing estimated class probabilities.
### Use VIF to clean matrix of class probabilities
### Evaluate ensemble of classifiers (LR, Penalized-LR, Stepwise, ...)
### Goal : Produce high accuracy ensemble classifiers from combinations of per-pixel classifiers.  Interpretability?
### Goal : Combine classifiers from multiple starting functional covariates (derivatives?  Benefits of per-pixel for non-functional data?)

setwd("C:\\Users\\rober\\Dropbox\\Arizona State University\\Ioannis Research\\Fall 2017 Projects\\Prospectus\\R Code and Figures\\Per-Pixel\\")
source("ESFuNC_Functions_March_2018.R")
#install.packages("fmsb")

#############################
### lupuskernelnorm dataset
#############################

lupuskernelnorm<-read.table("lupustherm.txt")
data.temp<-as.matrix(lupuskernelnorm[,1:451])
classes.temp<-as.numeric(lupuskernelnorm[,452])
argvals=seq(45, 90, 0.1)

nonSLE.cases<-which(classes.temp==0)
nonSLE.data<-data.temp[nonSLE.cases,]
SLE.cases<-which(classes.temp==1)
SLE.data<-data.temp[SLE.cases,]

### FULL BASIS NO SMOOTHING ###  UNSMOOTED in Manuscript
lupus.fdata<-fdata(data.temp, argvals=argvals)
lupus.fd<-fdata2fd(lupus.fdata, nbasis=451)
lupus.fdata<-fdata(t(eval.fd(argvals, lupus.fd)), argvals=argvals)

lupus.fd.d1<-fdata2fd(lupus.fdata, nbasis=451, nderiv=1)
lupus.fdata.d1<-fdata(t(eval.fd(argvals, lupus.fd.d1)), argvals=argvals)

lupus.fd.d2<-fdata2fd(lupus.fdata, nbasis=451, nderiv=2)
lupus.fdata.d2<-fdata(t(eval.fd(argvals, lupus.fd.d2)), argvals=argvals)

# ### REDUCED BASIS NO SMOOTHING ###  GCV in MANUSCRIPT
# lupus.fd.GCV<-fdata2fd(lupus.fdata, nbasis=150, lambda=0)
# lupus.fdata.GCV<-fdata(t(eval.fd(argvals, lupus.fd.GCV)), argvals=argvals)
# 
# lupus.fd.GCV.d1<-fdata2fd(lupus.fdata, nbasis=150, lambda=0, nderiv=1)
# lupus.fdata.GCV.d1<-fdata(t(eval.fd(argvals, lupus.fd.GCV.d1)), argvals=argvals)
# 
# lupus.fd.GCV.d2<-fdata2fd(lupus.fdata, nbasis=150, lambda=0, nderiv=2)
# lupus.fdata.GCV.d2<-fdata(t(eval.fd(argvals, lupus.fd.GCV.d2)), argvals=argvals)

### PACKAGES AND CROSS VALIDATION ###

library("glmnet")
library("MASS")
library("foreach")
library("doParallel")

### CV Settings ###
folds=10
trials=20
set.seed(1)
folds.list=fold.creator(classes.temp, folds, trials)

### Cores on ###
use.cores<-6
cl.temp<-makeCluster(use.cores)
registerDoParallel(cl.temp)

trials=2

### Calculate LOOCV class estimates from KNN
### Alternative classifiers? LR, discriminant, SVM, even possibly neural nets?
predictor.matrix.d0<-lupus.fdata$data
predictor.matrix.d1<-lupus.fdata.d1$data
predictor.matrix.d2<-lupus.fdata.d2$data
n.x<-length(argvals)
n.samples<-length(classes.temp)
k.npc.grid<-2:250
k.npc.size<-length(k.npc.grid)

output.d0<-foreach(k=1:n.x, .export="segment.class", .combine=append) %dopar%
{
  #cat(k, "\n")
  x<-predictor.matrix.d0[,k]
  dist.mat<-matrix(ncol=n.samples, nrow=n.samples)
  for(j in 1:n.samples) dist.mat[,j]<-sqrt((x[j]-x)^2)
  out<-segment.class(dist.mat, classes.temp, k.npc.grid, "wknn", kern.tri)
  output<-list(k.grid.accs=out$accuracy.est, maximum.accuracy=which.max(out$accuracy.est), best.acc.probabilities=out$prob.array[,2,which.max(out$accuracy.est)])
  return(list(output))
}

accuracies.perpixel.d0<-matrix(ncol=k.npc.size, nrow=n.x)
best.acc.k.perpixel.d0<-numeric()
best.acc.perpixel.d0<-numeric()
est.class.probs.perpixel.d0<-matrix(ncol=n.x, nrow=n.samples) #col=pixels, row=patients for LOOCV, getting each patients estimated class from LOOCV for every pixel!


for(j in 1:n.x)
{
  accuracies.perpixel.d0[j,]<-output.d0[[j]]$k.grid.accs
  best.acc.k.perpixel.d0[j]<-k.npc.grid[output.d0[[j]]$maximum.accuracy]
  best.acc.perpixel.d0[j]<-output.d0[[j]]$k.grid.accs[output.d0[[j]]$maximum.accuracy]
  est.class.probs.perpixel.d0[,j]<-output.d0[[j]]$best.acc.probabilities
}

save.image("Per-Pixel_SLE_Results.RData")

### D1 ####

output.d1<-foreach(k=1:n.x, .export="segment.class", .combine=append) %dopar%
{
  #cat(k, "\n")
  x<-predictor.matrix.d1[,k]
  dist.mat<-matrix(ncol=n.samples, nrow=n.samples)
  for(j in 1:n.samples) dist.mat[,j]<-sqrt((x[j]-x)^2)
  out<-segment.class(dist.mat, classes.temp, k.npc.grid, "wknn", kern.tri)
  output<-list(k.grid.accs=out$accuracy.est, maximum.accuracy=which.max(out$accuracy.est), best.acc.probabilities=out$prob.array[,2,which.max(out$accuracy.est)])
  return(list(output))
}

accuracies.perpixel.d1<-matrix(ncol=k.npc.size, nrow=n.x)
best.acc.k.perpixel.d1<-numeric()
best.acc.perpixel.d1<-numeric()
est.class.probs.perpixel.d1<-matrix(ncol=n.x, nrow=n.samples) #col=pixels, row=patients for LOOCV, getting each patients estimated class from LOOCV for every pixel!


for(j in 1:n.x)
{
  accuracies.perpixel.d1[j,]<-output.d1[[j]]$k.grid.accs
  best.acc.k.perpixel.d1[j]<-k.npc.grid[output.d1[[j]]$maximum.accuracy]
  best.acc.perpixel.d1[j]<-output.d1[[j]]$k.grid.accs[output.d1[[j]]$maximum.accuracy]
  est.class.probs.perpixel.d1[,j]<-output.d1[[j]]$best.acc.probabilities
}

save.image("Per-Pixel_SLE_Results.RData")

### D2 ####


output.d2<-foreach(k=1:n.x, .export="segment.class", .combine=append) %dopar%
{
  #cat(k, "\n")
  x<-predictor.matrix.d2[,k]
  dist.mat<-matrix(ncol=n.samples, nrow=n.samples)
  for(j in 1:n.samples) dist.mat[,j]<-sqrt((x[j]-x)^2)
  out<-segment.class(dist.mat, classes.temp, k.npc.grid, "wknn", kern.tri)
  output<-list(k.grid.accs=out$accuracy.est, maximum.accuracy=which.max(out$accuracy.est), best.acc.probabilities=out$prob.array[,2,which.max(out$accuracy.est)])
  return(list(output))
}

accuracies.perpixel.d2<-matrix(ncol=k.npc.size, nrow=n.x)
best.acc.k.perpixel.d2<-numeric()
best.acc.perpixel.d2<-numeric()
est.class.probs.perpixel.d2<-matrix(ncol=n.x, nrow=n.samples) #col=pixels, row=patients for LOOCV, getting each patients estimated class from LOOCV for every pixel!


for(j in 1:n.x)
{
  accuracies.perpixel.d2[j,]<-output.d2[[j]]$k.grid.accs
  best.acc.k.perpixel.d2[j]<-k.npc.grid[output.d2[[j]]$maximum.accuracy]
  best.acc.perpixel.d2[j]<-output.d2[[j]]$k.grid.accs[output.d2[[j]]$maximum.accuracy]
  est.class.probs.perpixel.d2[,j]<-output.d2[[j]]$best.acc.probabilities
}

save.image("Per-Pixel_SLE_Results.RData")

stopCluster(cl.temp)
rm(output.d0, output.d1, output.d2)

save.image("Per-Pixel_SLE_Results.RData")

source("part0_vif_func.R")
vif.d0.thresh.20<-vif_func(est.class.probs.perpixel.d0, thresh=20)
vif.d0.thresh.10<-vif_func(data.frame(est.class.probs.perpixel.d0)[,vif.d0.thresh.20], thresh=10)
vif.d0.thresh.5<-vif_func(data.frame(est.class.probs.perpixel.d0)[,vif.d0.thresh.10], thresh=5)

vif.d0.20.data<-as.matrix(data.frame(est.class.probs.perpixel.d0)[,vif.d0.thresh.20])
vif.d0.10.data<-as.matrix(data.frame(est.class.probs.perpixel.d0)[,vif.d0.thresh.10])
vif.d0.5.data<-as.matrix(data.frame(est.class.probs.perpixel.d0)[,vif.d0.thresh.5])

save.image("Per-Pixel_SLE_Results.RData")

vif.d1.thresh.20<-vif_func(est.class.probs.perpixel.d1, thresh=20)
vif.d1.thresh.10<-vif_func(data.frame(est.class.probs.perpixel.d1)[,vif.d1.thresh.20], thresh=10)
vif.d1.thresh.5<-vif_func(data.frame(est.class.probs.perpixel.d1)[,vif.d1.thresh.10], thresh=5)

vif.d1.20.data<-as.matrix(data.frame(est.class.probs.perpixel.d1)[,vif.d1.thresh.20])
vif.d1.10.data<-as.matrix(data.frame(est.class.probs.perpixel.d1)[,vif.d1.thresh.10])
vif.d1.5.data<-as.matrix(data.frame(est.class.probs.perpixel.d1)[,vif.d1.thresh.5])

save.image("Per-Pixel_SLE_Results.RData")

vif.d2.thresh.20<-vif_func(est.class.probs.perpixel.d2, thresh=20)
vif.d2.thresh.10<-vif_func(data.frame(est.class.probs.perpixel.d2)[,vif.d2.thresh.20], thresh=10)
vif.d2.thresh.5<-vif_func(data.frame(est.class.probs.perpixel.d2)[,vif.d2.thresh.10], thresh=5)

vif.d2.20.data<-as.matrix(data.frame(est.class.probs.perpixel.d2)[,vif.d2.thresh.20])
vif.d2.10.data<-as.matrix(data.frame(est.class.probs.perpixel.d2)[,vif.d2.thresh.10])
vif.d2.5.data<-as.matrix(data.frame(est.class.probs.perpixel.d2)[,vif.d2.thresh.5])

save.image("Per-Pixel_SLE_Results.RData")
save.image("Per-Pixel_SLE_Results_VIF_FINISHED.RData")

### Contempory Methods on PCA-predictors with Ensembles ###

load("Per-Pixel_SLE_Results_VIF_FINISHED.RData")

source("Supervised_Learning_Algorithms.R")
source("Ensemble_Functions.R")
use.cores<-6
cl.temp<-makeCluster(use.cores)
registerDoParallel(cl.temp)
trials<-2

equal.mat<-cbind(rep(1,folds*trials), rep(1,folds*trials), rep(1,folds*trials))

comb.d0.d1<-cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d1)
comb.d0.d2<-cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d2)
comb.d1.d2<-cbind(est.class.probs.perpixel.d1, est.class.probs.perpixel.d2)
comb.d0.d1.d2<-cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d1, est.class.probs.perpixel.d2)

comb.d0.d1.vif.20<-cbind(vif.d0.20.data, vif.d1.20.data)
comb.d0.d2.vif.20<<-cbind(vif.d0.20.data, vif.d2.20.data)
comb.d1.d2.vif.20<<-cbind(vif.d1.20.data, vif.d2.20.data)
comb.d0.d1.d2.vif.20<<-cbind(vif.d0.20.data, vif.d1.20.data, vif.d2.20.data)

comb.d0.d1.vif.10<-cbind(vif.d0.10.data, vif.d1.10.data)
comb.d0.d2.vif.10<<-cbind(vif.d0.10.data, vif.d2.10.data)
comb.d1.d2.vif.10<<-cbind(vif.d1.10.data, vif.d2.10.data)
comb.d0.d1.d2.vif.10<<-cbind(vif.d0.10.data, vif.d1.10.data, vif.d2.10.data)

comb.d0.d1.vif.5<-cbind(vif.d0.5.data, vif.d1.5.data)
comb.d0.d2.vif.5<<-cbind(vif.d0.5.data, vif.d2.5.data)
comb.d1.d2.vif.5<<-cbind(vif.d1.5.data, vif.d2.5.data)
comb.d0.d1.d2.vif.5<<-cbind(vif.d0.5.data, vif.d1.5.data, vif.d2.5.data)

### LR ###

### D0
lr.test.d0<-lr.kcv(t(est.class.probs.perpixel.d0), classes.temp, folds.list)
lr.test.d0.vif.20<-lr.kcv(t(vif.d0.20.data), classes.temp, folds.list)
lr.test.d0.vif.10<-lr.kcv(t(vif.d0.10.data), classes.temp, folds.list)
lr.test.d0.vif.5<-lr.kcv(t(vif.d0.5.data), classes.temp, folds.list)
### D1
lr.test.d1<-lr.kcv(t(est.class.probs.perpixel.d1), classes.temp, folds.list)
lr.test.d1.vif.20<-lr.kcv(t(vif.d0.20.data), classes.temp, folds.list)
lr.test.d1.vif.10<-lr.kcv(t(vif.d0.10.data), classes.temp, folds.list)
lr.test.d1.vif.5<-lr.kcv(t(vif.d0.5.data), classes.temp, folds.list)
### D2
lr.test.d2<-lr.kcv(t(est.class.probs.perpixel.d2), classes.temp, folds.list)
lr.test.d2.vif.20<-lr.kcv(t(vif.d0.20.data), classes.temp, folds.list)
lr.test.d2.vif.10<-lr.kcv(t(vif.d0.10.data), classes.temp, folds.list)
lr.test.d2.vif.5<-lr.kcv(t(vif.d0.5.data), classes.temp, folds.list)

### FOLD ACCURACIES IN MATRIX FORM FOR EACH DERIVATIVE ### USED AS WEIGHTS IN ENSEMBLES ###
lr.accuracy.mat<-cbind(lr.test.d0$accuracy, lr.test.d1$accuracy, lr.test.d2$accuracy)
lr.accuracy.mat.vif.20<-cbind(lr.test.d0.vif.20$accuracy, lr.test.d1.vif.20$accuracy, lr.test.d2.vif.20$accuracy)
lr.accuracy.mat.vif.10<-cbind(lr.test.d0.vif.10$accuracy, lr.test.d1.vif.10$accuracy, lr.test.d2.vif.10$accuracy)
lr.accuracy.mat.vif.5<-cbind(lr.test.d0.vif.5$accuracy, lr.test.d1.vif.5$accuracy, lr.test.d2.vif.5$accuracy)

### ESTIMATED CLASS PROBABILITIES IN A LIST ### USED IN ENSEMBLES ###
lr.est.class.list<-list(lr.test.d0$est.class.probs, lr.test.d1$est.class.probs, lr.test.d2$est.class.probs)
lr.est.class.list.vif.20<-list(lr.test.d0.vif.20$est.class.probs, lr.test.d1.vif.20$est.class.probs, lr.test.d2.vif.20$est.class.probs)
lr.est.class.list.vif.10<-list(lr.test.d0.vif.10$est.class.probs, lr.test.d1.vif.10$est.class.probs, lr.test.d2.vif.10$est.class.probs)
lr.est.class.list.vif.5<-list(lr.test.d0.vif.5$est.class.probs, lr.test.d1.vif.5$est.class.probs, lr.test.d2.vif.5$est.class.probs)

### NO VIF
lr.naive<-naive.ensembler.kcv(lr.est.class.list, classes.temp)
lr.acc.weight.2<-weighted.ensembler.kcv(lr.accuracy.mat, lr.est.class.list, 2, classes.temp)
lr.acc.weight.3<-weighted.ensembler.kcv(lr.accuracy.mat, lr.est.class.list, 3, classes.temp)
lr.equal.weight.2<-weighted.ensembler.kcv(equal.mat, lr.est.class.list, 2, classes.temp)
lr.equal.weight.3<-weighted.ensembler.kcv(equal.mat, lr.est.class.list, 3, classes.temp)

### VIF THRESH = 20 
lr.naive.vif.20<-naive.ensembler.kcv(lr.est.class.list.vif.20, classes.temp)
lr.acc.weight.2.vif.20<-weighted.ensembler.kcv(lr.accuracy.mat.vif.20, lr.est.class.list.vif.20, 2, classes.temp)
lr.acc.weight.3.vif.20<-weighted.ensembler.kcv(lr.accuracy.mat.vif.20, lr.est.class.list.vif.20, 3, classes.temp)
lr.equal.weight.2.vif.20<-weighted.ensembler.kcv(equal.mat, lr.est.class.list.vif.20, 2, classes.temp)
lr.equal.weight.3.vif.20<-weighted.ensembler.kcv(equal.mat, lr.est.class.list.vif.20, 3, classes.temp)

### VIF THRESH = 10 
lr.naive.vif.10<-naive.ensembler.kcv(lr.est.class.list.vif.10, classes.temp)
lr.acc.weight.2.vif.10<-weighted.ensembler.kcv(lr.accuracy.mat.vif.10, lr.est.class.list.vif.10, 2, classes.temp)
lr.acc.weight.3.vif.10<-weighted.ensembler.kcv(lr.accuracy.mat.vif.10, lr.est.class.list.vif.10, 3, classes.temp)
lr.equal.weight.2.vif.10<-weighted.ensembler.kcv(equal.mat, lr.est.class.list.vif.10, 2, classes.temp)
lr.equal.weight.3.vif.10<-weighted.ensembler.kcv(equal.mat, lr.est.class.list.vif.10, 3, classes.temp)

### VIF THRESH = 5 
lr.naive.vif.5<-naive.ensembler.kcv(lr.est.class.list.vif.5, classes.temp)
lr.acc.weight.2.vif.5<-weighted.ensembler.kcv(lr.accuracy.mat.vif.5, lr.est.class.list.vif.5, 2, classes.temp)
lr.acc.weight.3.vif.5<-weighted.ensembler.kcv(lr.accuracy.mat.vif.5, lr.est.class.list.vif.5, 3, classes.temp)
lr.equal.weight.2.vif.5<-weighted.ensembler.kcv(equal.mat, lr.est.class.list.vif.5, 2, classes.temp)
lr.equal.weight.3.vif.5<-weighted.ensembler.kcv(equal.mat, lr.est.class.list.vif.5, 3, classes.temp)

save.image("Per-Pixel_SLE_Results.RData")

# colMeans(lr.equal.weight.2.vif.20$accuracy)
# colMeans(lasso.equal.weight.2.vif.20$accuracy)
# colMeans(enet.equal.weight.2.vif.20$accuracy)
# colMeans(ridge.equal.weight.3$accuracy)
# colMeans(ridge.acc.weight.3$accuracy)
# colMeans(enet.acc.weight.3$accuracy)
# colMeans(lasso.acc.weight.3$accuracy)

### lasso ###
cat("lasso \n")

### D0
lasso.test.d0<-lasso.kcv(t(est.class.probs.perpixel.d0), classes.temp, folds.list)
lasso.test.d0.vif.20<-lasso.kcv(t(vif.d0.20.data), classes.temp, folds.list)
lasso.test.d0.vif.10<-lasso.kcv(t(vif.d0.10.data), classes.temp, folds.list)
lasso.test.d0.vif.5<-lasso.kcv(t(vif.d0.5.data), classes.temp, folds.list)
### D1
lasso.test.d1<-lasso.kcv(t(est.class.probs.perpixel.d1), classes.temp, folds.list)
lasso.test.d1.vif.20<-lasso.kcv(t(vif.d0.20.data), classes.temp, folds.list)
lasso.test.d1.vif.10<-lasso.kcv(t(vif.d0.10.data), classes.temp, folds.list)
lasso.test.d1.vif.5<-lasso.kcv(t(vif.d0.5.data), classes.temp, folds.list)
### D2
lasso.test.d2<-lasso.kcv(t(est.class.probs.perpixel.d2), classes.temp, folds.list)
lasso.test.d2.vif.20<-lasso.kcv(t(vif.d0.20.data), classes.temp, folds.list)
lasso.test.d2.vif.10<-lasso.kcv(t(vif.d0.10.data), classes.temp, folds.list)
lasso.test.d2.vif.5<-lasso.kcv(t(vif.d0.5.data), classes.temp, folds.list)

### FOLD ACCURACIES IN MATRIX FORM FOR EACH DERIVATIVE ### USED AS WEIGHTS IN ENSEMBLES ###
lasso.accuracy.mat<-cbind(lasso.test.d0$accuracy, lasso.test.d1$accuracy, lasso.test.d2$accuracy)
lasso.accuracy.mat.vif.20<-cbind(lasso.test.d0.vif.20$accuracy, lasso.test.d1.vif.20$accuracy, lasso.test.d2.vif.20$accuracy)
lasso.accuracy.mat.vif.10<-cbind(lasso.test.d0.vif.10$accuracy, lasso.test.d1.vif.10$accuracy, lasso.test.d2.vif.10$accuracy)
lasso.accuracy.mat.vif.5<-cbind(lasso.test.d0.vif.5$accuracy, lasso.test.d1.vif.5$accuracy, lasso.test.d2.vif.5$accuracy)

### ESTIMATED CLASS PROBABILITIES IN A LIST ### USED IN ENSEMBLES ###
lasso.est.class.list<-list(lasso.test.d0$est.class.probs, lasso.test.d1$est.class.probs, lasso.test.d2$est.class.probs)
lasso.est.class.list.vif.20<-list(lasso.test.d0.vif.20$est.class.probs, lasso.test.d1.vif.20$est.class.probs, lasso.test.d2.vif.20$est.class.probs)
lasso.est.class.list.vif.10<-list(lasso.test.d0.vif.10$est.class.probs, lasso.test.d1.vif.10$est.class.probs, lasso.test.d2.vif.10$est.class.probs)
lasso.est.class.list.vif.5<-list(lasso.test.d0.vif.5$est.class.probs, lasso.test.d1.vif.5$est.class.probs, lasso.test.d2.vif.5$est.class.probs)

### NO VIF
lasso.naive<-naive.ensembler.kcv(lasso.est.class.list, classes.temp)
lasso.acc.weight.2<-weighted.ensembler.kcv(lasso.accuracy.mat, lasso.est.class.list, 2, classes.temp)
lasso.acc.weight.3<-weighted.ensembler.kcv(lasso.accuracy.mat, lasso.est.class.list, 3, classes.temp)
lasso.equal.weight.2<-weighted.ensembler.kcv(equal.mat, lasso.est.class.list, 2, classes.temp)
lasso.equal.weight.3<-weighted.ensembler.kcv(equal.mat, lasso.est.class.list, 3, classes.temp)

### VIF THRESH = 20 
lasso.naive.vif.20<-naive.ensembler.kcv(lasso.est.class.list.vif.20, classes.temp)
lasso.acc.weight.2.vif.20<-weighted.ensembler.kcv(lasso.accuracy.mat.vif.20, lasso.est.class.list.vif.20, 2, classes.temp)
lasso.acc.weight.3.vif.20<-weighted.ensembler.kcv(lasso.accuracy.mat.vif.20, lasso.est.class.list.vif.20, 3, classes.temp)
lasso.equal.weight.2.vif.20<-weighted.ensembler.kcv(equal.mat, lasso.est.class.list.vif.20, 2, classes.temp)
lasso.equal.weight.3.vif.20<-weighted.ensembler.kcv(equal.mat, lasso.est.class.list.vif.20, 3, classes.temp)

### VIF THRESH = 10 
lasso.naive.vif.10<-naive.ensembler.kcv(lasso.est.class.list.vif.10, classes.temp)
lasso.acc.weight.2.vif.10<-weighted.ensembler.kcv(lasso.accuracy.mat.vif.10, lasso.est.class.list.vif.10, 2, classes.temp)
lasso.acc.weight.3.vif.10<-weighted.ensembler.kcv(lasso.accuracy.mat.vif.10, lasso.est.class.list.vif.10, 3, classes.temp)
lasso.equal.weight.2.vif.10<-weighted.ensembler.kcv(equal.mat, lasso.est.class.list.vif.10, 2, classes.temp)
lasso.equal.weight.3.vif.10<-weighted.ensembler.kcv(equal.mat, lasso.est.class.list.vif.10, 3, classes.temp)

### VIF THRESH = 5 
lasso.naive.vif.5<-naive.ensembler.kcv(lasso.est.class.list.vif.5, classes.temp)
lasso.acc.weight.2.vif.5<-weighted.ensembler.kcv(lasso.accuracy.mat.vif.5, lasso.est.class.list.vif.5, 2, classes.temp)
lasso.acc.weight.3.vif.5<-weighted.ensembler.kcv(lasso.accuracy.mat.vif.5, lasso.est.class.list.vif.5, 3, classes.temp)
lasso.equal.weight.2.vif.5<-weighted.ensembler.kcv(equal.mat, lasso.est.class.list.vif.5, 2, classes.temp)
lasso.equal.weight.3.vif.5<-weighted.ensembler.kcv(equal.mat, lasso.est.class.list.vif.5, 3, classes.temp)

### COMBINED FITS AT DIFFERENT VIF THRESH ###

### D0 D1
lasso.test.d0.d1<-lasso.kcv(t(comb.d0.d1), classes.temp, folds.list)
lasso.test.d0.d1.vif.20<-lasso.kcv(t(comb.d0.d1.vif.20), classes.temp, folds.list)
lasso.test.d0.d1.vif.10<-lasso.kcv(t(comb.d0.d1.vif.10), classes.temp, folds.list)
lasso.test.d0.d1.vif.5<-lasso.kcv(t(comb.d0.d1.vif.5), classes.temp, folds.list)

### D0 D2
lasso.test.d0.d2<-lasso.kcv(t(comb.d0.d2), classes.temp, folds.list)
lasso.test.d0.d2.vif.20<-lasso.kcv(t(comb.d0.d2.vif.20), classes.temp, folds.list)
lasso.test.d0.d2.vif.10<-lasso.kcv(t(comb.d0.d2.vif.10), classes.temp, folds.list)
lasso.test.d0.d2.vif.5<-lasso.kcv(t(comb.d0.d2.vif.5), classes.temp, folds.list)

### D1 D2
lasso.test.d1.d2<-lasso.kcv(t(comb.d1.d2), classes.temp, folds.list)
lasso.test.d1.d2.vif.20<-lasso.kcv(t(comb.d1.d2.vif.20), classes.temp, folds.list)
lasso.test.d1.d2.vif.10<-lasso.kcv(t(comb.d1.d2.vif.10), classes.temp, folds.list)
lasso.test.d1.d2.vif.5<-lasso.kcv(t(comb.d1.d2.vif.5), classes.temp, folds.list)

### D0 D1 D2
lasso.test.d0.d1.d2<-lasso.kcv(t(comb.d0.d1.d2), classes.temp, folds.list)
lasso.test.d0.d1.d2.vif.20<-lasso.kcv(t(comb.d0.d1.d2.vif.20), classes.temp, folds.list)
lasso.test.d0.d1.d2.vif.10<-lasso.kcv(t(comb.d0.d1.d2.vif.10), classes.temp, folds.list)
lasso.test.d0.d1.d2.vif.5<-lasso.kcv(t(comb.d0.d1.d2.vif.5), classes.temp, folds.list)

save.image("Per-Pixel_SLE_Results.RData")

### enet ###

### D0
enet.test.d0<-enet.kcv(t(est.class.probs.perpixel.d0), classes.temp, folds.list)
enet.test.d0.vif.20<-enet.kcv(t(vif.d0.20.data), classes.temp, folds.list)
enet.test.d0.vif.10<-enet.kcv(t(vif.d0.10.data), classes.temp, folds.list)
enet.test.d0.vif.5<-enet.kcv(t(vif.d0.5.data), classes.temp, folds.list)
### D1
enet.test.d1<-enet.kcv(t(est.class.probs.perpixel.d1), classes.temp, folds.list)
enet.test.d1.vif.20<-enet.kcv(t(vif.d0.20.data), classes.temp, folds.list)
enet.test.d1.vif.10<-enet.kcv(t(vif.d0.10.data), classes.temp, folds.list)
enet.test.d1.vif.5<-enet.kcv(t(vif.d0.5.data), classes.temp, folds.list)
### D2
enet.test.d2<-enet.kcv(t(est.class.probs.perpixel.d2), classes.temp, folds.list)
enet.test.d2.vif.20<-enet.kcv(t(vif.d0.20.data), classes.temp, folds.list)
enet.test.d2.vif.10<-enet.kcv(t(vif.d0.10.data), classes.temp, folds.list)
enet.test.d2.vif.5<-enet.kcv(t(vif.d0.5.data), classes.temp, folds.list)

### FOLD ACCURACIES IN MATRIX FORM FOR EACH DERIVATIVE ### USED AS WEIGHTS IN ENSEMBLES ###
enet.accuracy.mat<-cbind(enet.test.d0$accuracy, enet.test.d1$accuracy, enet.test.d2$accuracy)
enet.accuracy.mat.vif.20<-cbind(enet.test.d0.vif.20$accuracy, enet.test.d1.vif.20$accuracy, enet.test.d2.vif.20$accuracy)
enet.accuracy.mat.vif.10<-cbind(enet.test.d0.vif.10$accuracy, enet.test.d1.vif.10$accuracy, enet.test.d2.vif.10$accuracy)
enet.accuracy.mat.vif.5<-cbind(enet.test.d0.vif.5$accuracy, enet.test.d1.vif.5$accuracy, enet.test.d2.vif.5$accuracy)

### ESTIMATED CLASS PROBABILITIES IN A LIST ### USED IN ENSEMBLES ###
enet.est.class.list<-list(enet.test.d0$est.class.probs, enet.test.d1$est.class.probs, enet.test.d2$est.class.probs)
enet.est.class.list.vif.20<-list(enet.test.d0.vif.20$est.class.probs, enet.test.d1.vif.20$est.class.probs, enet.test.d2.vif.20$est.class.probs)
enet.est.class.list.vif.10<-list(enet.test.d0.vif.10$est.class.probs, enet.test.d1.vif.10$est.class.probs, enet.test.d2.vif.10$est.class.probs)
enet.est.class.list.vif.5<-list(enet.test.d0.vif.5$est.class.probs, enet.test.d1.vif.5$est.class.probs, enet.test.d2.vif.5$est.class.probs)

### NO VIF
enet.naive<-naive.ensembler.kcv(enet.est.class.list, classes.temp)
enet.acc.weight.2<-weighted.ensembler.kcv(enet.accuracy.mat, enet.est.class.list, 2, classes.temp)
enet.acc.weight.3<-weighted.ensembler.kcv(enet.accuracy.mat, enet.est.class.list, 3, classes.temp)
enet.equal.weight.2<-weighted.ensembler.kcv(equal.mat, enet.est.class.list, 2, classes.temp)
enet.equal.weight.3<-weighted.ensembler.kcv(equal.mat, enet.est.class.list, 3, classes.temp)

### VIF THRESH = 20 
enet.naive.vif.20<-naive.ensembler.kcv(enet.est.class.list.vif.20, classes.temp)
enet.acc.weight.2.vif.20<-weighted.ensembler.kcv(enet.accuracy.mat.vif.20, enet.est.class.list.vif.20, 2, classes.temp)
enet.acc.weight.3.vif.20<-weighted.ensembler.kcv(enet.accuracy.mat.vif.20, enet.est.class.list.vif.20, 3, classes.temp)
enet.equal.weight.2.vif.20<-weighted.ensembler.kcv(equal.mat, enet.est.class.list.vif.20, 2, classes.temp)
enet.equal.weight.3.vif.20<-weighted.ensembler.kcv(equal.mat, enet.est.class.list.vif.20, 3, classes.temp)

### VIF THRESH = 10 
enet.naive.vif.10<-naive.ensembler.kcv(enet.est.class.list.vif.10, classes.temp)
enet.acc.weight.2.vif.10<-weighted.ensembler.kcv(enet.accuracy.mat.vif.10, enet.est.class.list.vif.10, 2, classes.temp)
enet.acc.weight.3.vif.10<-weighted.ensembler.kcv(enet.accuracy.mat.vif.10, enet.est.class.list.vif.10, 3, classes.temp)
enet.equal.weight.2.vif.10<-weighted.ensembler.kcv(equal.mat, enet.est.class.list.vif.10, 2, classes.temp)
enet.equal.weight.3.vif.10<-weighted.ensembler.kcv(equal.mat, enet.est.class.list.vif.10, 3, classes.temp)

### VIF THRESH = 5 
enet.naive.vif.5<-naive.ensembler.kcv(enet.est.class.list.vif.5, classes.temp)
enet.acc.weight.2.vif.5<-weighted.ensembler.kcv(enet.accuracy.mat.vif.5, enet.est.class.list.vif.5, 2, classes.temp)
enet.acc.weight.3.vif.5<-weighted.ensembler.kcv(enet.accuracy.mat.vif.5, enet.est.class.list.vif.5, 3, classes.temp)
enet.equal.weight.2.vif.5<-weighted.ensembler.kcv(equal.mat, enet.est.class.list.vif.5, 2, classes.temp)
enet.equal.weight.3.vif.5<-weighted.ensembler.kcv(equal.mat, enet.est.class.list.vif.5, 3, classes.temp)

### COMBINED FITS AT DIFFERENT VIF THRESH ###

### D0 D1
enet.test.d0.d1<-enet.kcv(t(comb.d0.d1), classes.temp, folds.list)
enet.test.d0.d1.vif.20<-enet.kcv(t(comb.d0.d1.vif.20), classes.temp, folds.list)
enet.test.d0.d1.vif.10<-enet.kcv(t(comb.d0.d1.vif.10), classes.temp, folds.list)
enet.test.d0.d1.vif.5<-enet.kcv(t(comb.d0.d1.vif.5), classes.temp, folds.list)

### D0 D2
enet.test.d0.d2<-enet.kcv(t(comb.d0.d2), classes.temp, folds.list)
enet.test.d0.d2.vif.20<-enet.kcv(t(comb.d0.d2.vif.20), classes.temp, folds.list)
enet.test.d0.d2.vif.10<-enet.kcv(t(comb.d0.d2.vif.10), classes.temp, folds.list)
enet.test.d0.d2.vif.5<-enet.kcv(t(comb.d0.d2.vif.5), classes.temp, folds.list)

### D1 D2
enet.test.d1.d2<-enet.kcv(t(comb.d1.d2), classes.temp, folds.list)
enet.test.d1.d2.vif.20<-enet.kcv(t(comb.d1.d2.vif.20), classes.temp, folds.list)
enet.test.d1.d2.vif.10<-enet.kcv(t(comb.d1.d2.vif.10), classes.temp, folds.list)
enet.test.d1.d2.vif.5<-enet.kcv(t(comb.d1.d2.vif.5), classes.temp, folds.list)

### D0 D1 D2
enet.test.d0.d1.d2<-enet.kcv(t(comb.d0.d1.d2), classes.temp, folds.list)
enet.test.d0.d1.d2.vif.20<-enet.kcv(t(comb.d0.d1.d2.vif.20), classes.temp, folds.list)
enet.test.d0.d1.d2.vif.10<-enet.kcv(t(comb.d0.d1.d2.vif.10), classes.temp, folds.list)
enet.test.d0.d1.d2.vif.5<-enet.kcv(t(comb.d0.d1.d2.vif.5), classes.temp, folds.list)

save.image("Per-Pixel_SLE_Results.RData")

### ridge ###

### D0
ridge.test.d0<-ridge.kcv(t(est.class.probs.perpixel.d0), classes.temp, folds.list)
ridge.test.d0.vif.20<-ridge.kcv(t(vif.d0.20.data), classes.temp, folds.list)
ridge.test.d0.vif.10<-ridge.kcv(t(vif.d0.10.data), classes.temp, folds.list)
ridge.test.d0.vif.5<-ridge.kcv(t(vif.d0.5.data), classes.temp, folds.list)
### D1
ridge.test.d1<-ridge.kcv(t(est.class.probs.perpixel.d1), classes.temp, folds.list)
ridge.test.d1.vif.20<-ridge.kcv(t(vif.d0.20.data), classes.temp, folds.list)
ridge.test.d1.vif.10<-ridge.kcv(t(vif.d0.10.data), classes.temp, folds.list)
ridge.test.d1.vif.5<-ridge.kcv(t(vif.d0.5.data), classes.temp, folds.list)
### D2
ridge.test.d2<-ridge.kcv(t(est.class.probs.perpixel.d2), classes.temp, folds.list)
ridge.test.d2.vif.20<-ridge.kcv(t(vif.d0.20.data), classes.temp, folds.list)
ridge.test.d2.vif.10<-ridge.kcv(t(vif.d0.10.data), classes.temp, folds.list)
ridge.test.d2.vif.5<-ridge.kcv(t(vif.d0.5.data), classes.temp, folds.list)

### FOLD ACCURACIES IN MATRIX FORM FOR EACH DERIVATIVE ### USED AS WEIGHTS IN ENSEMBLES ###
ridge.accuracy.mat<-cbind(ridge.test.d0$accuracy, ridge.test.d1$accuracy, ridge.test.d2$accuracy)
ridge.accuracy.mat.vif.20<-cbind(ridge.test.d0.vif.20$accuracy, ridge.test.d1.vif.20$accuracy, ridge.test.d2.vif.20$accuracy)
ridge.accuracy.mat.vif.10<-cbind(ridge.test.d0.vif.10$accuracy, ridge.test.d1.vif.10$accuracy, ridge.test.d2.vif.10$accuracy)
ridge.accuracy.mat.vif.5<-cbind(ridge.test.d0.vif.5$accuracy, ridge.test.d1.vif.5$accuracy, ridge.test.d2.vif.5$accuracy)

### ESTIMATED CLASS PROBABILITIES IN A LIST ### USED IN ENSEMBLES ###
ridge.est.class.list<-list(ridge.test.d0$est.class.probs, ridge.test.d1$est.class.probs, ridge.test.d2$est.class.probs)
ridge.est.class.list.vif.20<-list(ridge.test.d0.vif.20$est.class.probs, ridge.test.d1.vif.20$est.class.probs, ridge.test.d2.vif.20$est.class.probs)
ridge.est.class.list.vif.10<-list(ridge.test.d0.vif.10$est.class.probs, ridge.test.d1.vif.10$est.class.probs, ridge.test.d2.vif.10$est.class.probs)
ridge.est.class.list.vif.5<-list(ridge.test.d0.vif.5$est.class.probs, ridge.test.d1.vif.5$est.class.probs, ridge.test.d2.vif.5$est.class.probs)

### NO VIF
ridge.naive<-naive.ensembler.kcv(ridge.est.class.list, classes.temp)
ridge.acc.weight.2<-weighted.ensembler.kcv(ridge.accuracy.mat, ridge.est.class.list, 2, classes.temp)
ridge.acc.weight.3<-weighted.ensembler.kcv(ridge.accuracy.mat, ridge.est.class.list, 3, classes.temp)
ridge.equal.weight.2<-weighted.ensembler.kcv(equal.mat, ridge.est.class.list, 2, classes.temp)
ridge.equal.weight.3<-weighted.ensembler.kcv(equal.mat, ridge.est.class.list, 3, classes.temp)

### VIF THRESH = 20 
ridge.naive.vif.20<-naive.ensembler.kcv(ridge.est.class.list.vif.20, classes.temp)
ridge.acc.weight.2.vif.20<-weighted.ensembler.kcv(ridge.accuracy.mat.vif.20, ridge.est.class.list.vif.20, 2, classes.temp)
ridge.acc.weight.3.vif.20<-weighted.ensembler.kcv(ridge.accuracy.mat.vif.20, ridge.est.class.list.vif.20, 3, classes.temp)
ridge.equal.weight.2.vif.20<-weighted.ensembler.kcv(equal.mat, ridge.est.class.list.vif.20, 2, classes.temp)
ridge.equal.weight.3.vif.20<-weighted.ensembler.kcv(equal.mat, ridge.est.class.list.vif.20, 3, classes.temp)

### VIF THRESH = 10 
ridge.naive.vif.10<-naive.ensembler.kcv(ridge.est.class.list.vif.10, classes.temp)
ridge.acc.weight.2.vif.10<-weighted.ensembler.kcv(ridge.accuracy.mat.vif.10, ridge.est.class.list.vif.10, 2, classes.temp)
ridge.acc.weight.3.vif.10<-weighted.ensembler.kcv(ridge.accuracy.mat.vif.10, ridge.est.class.list.vif.10, 3, classes.temp)
ridge.equal.weight.2.vif.10<-weighted.ensembler.kcv(equal.mat, ridge.est.class.list.vif.10, 2, classes.temp)
ridge.equal.weight.3.vif.10<-weighted.ensembler.kcv(equal.mat, ridge.est.class.list.vif.10, 3, classes.temp)

### VIF THRESH = 5 
ridge.naive.vif.5<-naive.ensembler.kcv(ridge.est.class.list.vif.5, classes.temp)
ridge.acc.weight.2.vif.5<-weighted.ensembler.kcv(ridge.accuracy.mat.vif.5, ridge.est.class.list.vif.5, 2, classes.temp)
ridge.acc.weight.3.vif.5<-weighted.ensembler.kcv(ridge.accuracy.mat.vif.5, ridge.est.class.list.vif.5, 3, classes.temp)
ridge.equal.weight.2.vif.5<-weighted.ensembler.kcv(equal.mat, ridge.est.class.list.vif.5, 2, classes.temp)
ridge.equal.weight.3.vif.5<-weighted.ensembler.kcv(equal.mat, ridge.est.class.list.vif.5, 3, classes.temp)

### COMBINED FITS AT DIFFERENT VIF THRESH ###

### D0 D1
ridge.test.d0.d1<-ridge.kcv(t(comb.d0.d1), classes.temp, folds.list)
ridge.test.d0.d1.vif.20<-ridge.kcv(t(comb.d0.d1.vif.20), classes.temp, folds.list)
ridge.test.d0.d1.vif.10<-ridge.kcv(t(comb.d0.d1.vif.10), classes.temp, folds.list)
ridge.test.d0.d1.vif.5<-ridge.kcv(t(comb.d0.d1.vif.5), classes.temp, folds.list)

### D0 D2
ridge.test.d0.d2<-ridge.kcv(t(comb.d0.d2), classes.temp, folds.list)
ridge.test.d0.d2.vif.20<-ridge.kcv(t(comb.d0.d2.vif.20), classes.temp, folds.list)
ridge.test.d0.d2.vif.10<-ridge.kcv(t(comb.d0.d2.vif.10), classes.temp, folds.list)
ridge.test.d0.d2.vif.5<-ridge.kcv(t(comb.d0.d2.vif.5), classes.temp, folds.list)

### D1 D2
ridge.test.d1.d2<-ridge.kcv(t(comb.d1.d2), classes.temp, folds.list)
ridge.test.d1.d2.vif.20<-ridge.kcv(t(comb.d1.d2.vif.20), classes.temp, folds.list)
ridge.test.d1.d2.vif.10<-ridge.kcv(t(comb.d1.d2.vif.10), classes.temp, folds.list)
ridge.test.d1.d2.vif.5<-ridge.kcv(t(comb.d1.d2.vif.5), classes.temp, folds.list)

### D0 D1 D2
ridge.test.d0.d1.d2<-ridge.kcv(t(comb.d0.d1.d2), classes.temp, folds.list)
ridge.test.d0.d1.d2.vif.20<-ridge.kcv(t(comb.d0.d1.d2.vif.20), classes.temp, folds.list)
ridge.test.d0.d1.d2.vif.10<-ridge.kcv(t(comb.d0.d1.d2.vif.10), classes.temp, folds.list)
ridge.test.d0.d1.d2.vif.5<-ridge.kcv(t(comb.d0.d1.d2.vif.5), classes.temp, folds.list)

save.image("Per-Pixel_SLE_Results.RData")













# which(best.acc.perpixel.d0==max(best.acc.perpixel.d0))
# best.acc.perpixel.d0[207]
# argvals[c(207,208)]
# which(best.acc.perpixel.d1==max(best.acc.perpixel.d1))
# argvals[c(231)]
# best.acc.perpixel.d1[231]
# which(best.acc.perpixel.d2==max(best.acc.perpixel.d2))
# argvals[c(262)]
# best.acc.perpixel.d1[262]
# 
# 
# 
# boxplot(accuracy.pc.kcv.d0, xlab="", col="grey75", xaxt="n", cex.axis=1.1, ylab="", yaxt="n", cex.lab=1.15, xlim=c(1,151))
# abline(v=best.d0.lr, col="grey50", lwd=3)
# lines(colMeans(accuracy.pc.kcv.d0), col="grey50", lwd=4)
# axis(side=2, at=seq(0.7,0.9,0.1), las=0, cex.axis=1, font=2)
# 
# plot(argvals, best.acc.perpixel.d0)
# 
# pdf("test.pdf")
# for(j in 1:451) 
# {
#   plot(est.class.probs.perpixel.d0[,j])
#   abline(h=0.5)
# }
# dev.off()
# 
# plot(est.class.probs.perpixel.d0[,1])
# for(j in 2:25) 
# {
#   points(est.class.probs.perpixel.d0[,j])
# }
# abline(h=0.5)
# 
# est.cl
# 
# 
# library(corrplot)
# 
# cor.d0<-cor(predictor.matrix.d0)
# cor.pp.d0<-cor(est.class.probs.perpixel.d0)
# cor.d1<-cor(predictor.matrix.d1)
# cor.pp.d1<-cor(est.class.probs.perpixel.d1)
# cor.d2<-cor(predictor.matrix.d2)
# cor.pp.d2<-cor(est.class.probs.perpixel.d2)
# 
# cor.d0.d1.d2<-cor(cbind(predictor.matrix.d0, predictor.matrix.d1, predictor.matrix.d2))
# cor.pp.d0.d1.d2<-cor(cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d1, est.class.probs.perpixel.d2))
# 
# tiff("corrtest.tiff")
# corrplot(cor.d0.d1.d2, method="color", tl.pos="n", type="upper")
# corrplot(cor.pp.d0.d1.d2, method="color", tl.pos="n", cl.pos="n", type="lower", add=TRUE)
# dev.off()
# 
# 
# 
# 
# 
# 
alpha<-0.5
est.classes.d0<-matrix(0, nrow=n.samples, ncol=n.x)
est.classes.d1<-matrix(0, nrow=n.samples, ncol=n.x)
est.classes.d2<-matrix(0, nrow=n.samples, ncol=n.x)
for(j in 1:n.x)
{
  est.classes.d0[est.class.probs.perpixel.d0[,j]>alpha,j]<-1
  est.classes.d1[est.class.probs.perpixel.d1[,j]>alpha,j]<-1
  est.classes.d2[est.class.probs.perpixel.d2[,j]>alpha,j]<-1
}

table(est.classes.d0[588,])
plot(est.classes.d0[588,])

t.1<-as.vector(table(est.classes.d0[1,]))
for(j in 2:589) t.1<-rbind(t.1, as.vector(table(est.classes.d0[j,])))
cbind(t.1, t.1[,2]/451)
# 
naive.vote<-t.1[,2]/451
# plot(naive.vote)
# abline(h=0.5)
# 
# naive.pred<-rep(0,589)
# naive.pred[which(naive.vote>0.5)]<-1
# plot(naive.pred)
# table(naive.pred, classes.temp)
# mean(naive.pred==classes.temp)
# 
# plot(est.class.probs.perpixel.d0[1,])
# mean(est.class.probs.perpixel.d0[1,])
# rowMeans(est.class.probs.perpixel.d0)
# 
# prob.pred<-rep(0,589)
# prob.pred[which(rowMeans(est.class.probs.perpixel.d0)>0.5)]<-1
# table(prob.pred, classes.temp)
# mean(prob.pred==classes.temp)
# 
# which.max(best.acc.perpixel.d0)
# 
# plot(est.class.probs.perpixel.d0[,207])
# 
# # 
# 
# 
# 
# 
# ### Analysis of Estimated Class Probabilities ###
# 
# # load("Per-Pixel_SLE.RData")
# # 
# # plot(best.acc.perpixel.d0)
# # plot(argvals, best.acc.perpixel.d1)
# # plot(argvals, best.acc.perpixel.d2)
# # 
# # plot(argvals, best.acc.k.perpixel.d0)
# # plot(argvals, best.acc.k.perpixel.d1)
# # plot(argvals, best.acc.k.perpixel.d2)
# # 
# # 
# # cor.numeric.d0<-cor(est.class.probs.perpixel.d0)
# # cor.numeric.d1<-cor(est.class.probs.perpixel.d1)
# # cor.numeric.d2<-cor(est.class.probs.perpixel.d2)
# 
# # library(corrplot)
# # pdf("corrplots.pdf")
# # corrplot(cor.numeric.d0, method="color", tl.pos="n")
# # corrplot(cor.numeric.d1, method="color", tl.pos="n")
# # corrplot(cor.numeric.d2, method="color", tl.pos="n")
# # dev.off()
# 
# # cor.numeric.d0.d1<-cor(cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d1))
# # cor.numeric.d0.d2<-cor(cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d2))
# # cor.numeric.d1.d2<-cor(cbind(est.class.probs.perpixel.d1, est.class.probs.perpixel.d2))
# # cor.numeric.d0.d1.d2<-cor(cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d1, est.class.probs.perpixel.d2))
# # library(corrplot)
# # tiff("corrplots_multivariate_d0d1.tiff")
# # corrplot(cor.numeric.d0.d1, method="color", tl.pos="n")
# # dev.off()
# # tiff("corrplots_multivariate_d0d2.tiff")
# # corrplot(cor.numeric.d0.d2, method="color", tl.pos="n")
# # dev.off()
# # tiff("corrplots_multivariate_d1d2.tiff")
# # corrplot(cor.numeric.d1.d2, method="color", tl.pos="n")
# # dev.off()
# # tiff("corrplots_multivariate_d0d1d2.tiff")
# # corrplot(cor.numeric.d0.d1.d2, method="color", tl.pos="n")
# # dev.off()
# # 
# # 
# alpha<-0.5
# est.classes.d0<-matrix(0, nrow=n.samples, ncol=n.x)
# est.classes.d1<-matrix(0, nrow=n.samples, ncol=n.x)
# est.classes.d2<-matrix(0, nrow=n.samples, ncol=n.x)
# for(j in 1:n.x)
# {
#   est.classes.d0[est.class.probs.perpixel.d0[,j]>alpha,j]<-1
#   est.classes.d1[est.class.probs.perpixel.d1[,j]>alpha,j]<-1
#   est.classes.d2[est.class.probs.perpixel.d2[,j]>alpha,j]<-1
# }
# # 
# # 
# # 
# # 
# # pdf("PerPixelTest_Plots.pdf")
# # for(k in 1:451) plot(accuracies.perpixel.d0[k,])
# # dev.off()
# 
# save.image("Per-Pixel_SLE.RData")
# 
# ### STEPWISE ENSEMBLERS ###
# 
# est.prob.array.d0<-array(dim=c(589,2,1,451))
# est.prob.array.d1<-array(dim=c(589,2,1,451))
# est.prob.array.d2<-array(dim=c(589,2,1,451))
# est.prob.array.d0.d1<-array(dim=c(589,2,1,902))
# est.prob.array.d0.d2<-array(dim=c(589,2,1,902))
# est.prob.array.d1.d2<-array(dim=c(589,2,1,902))
# est.prob.array.d0.d1.d2<-array(dim=c(589,2,1,1353))
# for(j in 1:n.x) est.prob.array.d0[,,1,j]<-cbind(1-est.class.probs.perpixel.d0[,j], est.class.probs.perpixel.d0[,j])
# for(j in 1:n.x) est.prob.array.d1[,,1,j]<-cbind(1-est.class.probs.perpixel.d1[,j], est.class.probs.perpixel.d1[,j])
# for(j in 1:n.x) est.prob.array.d2[,,1,j]<-cbind(1-est.class.probs.perpixel.d2[,j], est.class.probs.perpixel.d2[,j])
# 
# for(j in 1:(2*n.x)) est.prob.array.d0.d1[,,1,j]<-cbind(1-cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d1)[,j], cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d1)[,j])
# for(j in 1:(2*n.x)) est.prob.array.d0.d2[,,1,j]<-cbind(1-cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d2)[,j], cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d2)[,j])
# for(j in 1:(2*n.x)) est.prob.array.d1.d2[,,1,j]<-cbind(1-cbind(est.class.probs.perpixel.d1, est.class.probs.perpixel.d2)[,j], cbind(est.class.probs.perpixel.d1, est.class.probs.perpixel.d2)[,j])
# 
# for(j in 1:(3*n.x)) est.prob.array.d0.d1.d2[,,1,j]<-cbind(1-cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d1, est.class.probs.perpixel.d2)[,j], cbind(est.class.probs.perpixel.d0, est.class.probs.perpixel.d1, est.class.probs.perpixel.d2)[,j])
# 
# 
# step.d0<-forward.ensemble(step.array = est.prob.array.d0, segment.accuracies = t(as.matrix(best.acc.perpixel.d0)), classes=classes.temp, seg.weight = FALSE, thresh = 0.00001, do.par = FALSE, cores=6)
# step.d0.weight<-forward.ensemble(step.array = est.prob.array.d0, segment.accuracies = t(as.matrix(best.acc.perpixel.d0)), classes=classes.temp, seg.weight = TRUE, thresh = 0.00001, do.par = FALSE, cores=6)
# 
# step.d1<-forward.ensemble(step.array = est.prob.array.d1, segment.accuracies = t(as.matrix(best.acc.perpixel.d1)), classes=classes.temp, seg.weight = FALSE, thresh = 0.00001, do.par = FALSE, cores=6)
# step.d1.weight<-forward.ensemble(step.array = est.prob.array.d1, segment.accuracies = t(as.matrix(best.acc.perpixel.d1)), classes=classes.temp, seg.weight = TRUE, thresh = 0.00001, do.par = FALSE, cores=6)
# 
# step.d2<-forward.ensemble(step.array = est.prob.array.d2, segment.accuracies = t(as.matrix(best.acc.perpixel.d2)), classes=classes.temp, seg.weight = FALSE, thresh = 0.00001, do.par = FALSE, cores=6)
# step.d2.weight<-forward.ensemble(step.array = est.prob.array.d2, segment.accuracies = t(as.matrix(best.acc.perpixel.d2)), classes=classes.temp, seg.weight = TRUE, thresh = 0.00001, do.par = FALSE, cores=6)
# 
# step.d0$ens.accuracies
# step.d0$ens.segments
# step.d0.weight$ens.accuracies
# step.d0.weight$ens.segments
# 
# step.d1$ens.accuracies
# step.d1$ens.segments
# step.d1.weight$ens.accuracies
# step.d1.weight$ens.segments
# 
# step.d2$ens.accuracies
# step.d2$ens.segments
# step.d2.weight$ens.accuracies
# step.d2.weight$ens.segments
# 
# step.d0.d1<-forward.ensemble(step.array = est.prob.array.d0.d1, segment.accuracies = t(as.matrix(c(best.acc.perpixel.d0, best.acc.perpixel.d1))), classes=classes.temp, seg.weight = FALSE, thresh = 0.00001, do.par = FALSE, cores=6)
# step.d0.d1.weight<-forward.ensemble(step.array = est.prob.array.d0.d1, segment.accuracies = t(as.matrix(c(best.acc.perpixel.d0, best.acc.perpixel.d1))), classes=classes.temp, seg.weight = TRUE, thresh = 0.00001, do.par = FALSE, cores=6)
# 
# step.d0.d1$ens.accuracies
# step.d0.d1$ens.segments
# step.d0.d1.weight$ens.accuracies
# step.d0.d1.weight$ens.segments
# 
# step.d0.d2<-forward.ensemble(step.array = est.prob.array.d0.d2, segment.accuracies = t(as.matrix(c(best.acc.perpixel.d0, best.acc.perpixel.d2))), classes=classes.temp, seg.weight = FALSE, thresh = 0.00001, do.par = FALSE, cores=6)
# step.d0.d2.weight<-forward.ensemble(step.array = est.prob.array.d0.d2, segment.accuracies = t(as.matrix(c(best.acc.perpixel.d0, best.acc.perpixel.d2))), classes=classes.temp, seg.weight = TRUE, thresh = 0.00001, do.par = FALSE, cores=6)
# 
# step.d0.d2$ens.accuracies
# step.d0.d2$ens.segments
# step.d0.d2.weight$ens.accuracies
# step.d0.d2.weight$ens.segments
# 
# step.d1.d2<-forward.ensemble(step.array = est.prob.array.d1.d2, segment.accuracies = t(as.matrix(c(best.acc.perpixel.d1, best.acc.perpixel.d2))), classes=classes.temp, seg.weight = FALSE, thresh = 0.00001, do.par = FALSE, cores=6)
# step.d1.d2.weight<-forward.ensemble(step.array = est.prob.array.d1.d2, segment.accuracies = t(as.matrix(c(best.acc.perpixel.d1, best.acc.perpixel.d2))), classes=classes.temp, seg.weight = TRUE, thresh = 0.00001, do.par = FALSE, cores=6)
# 
# step.d1.d2$ens.accuracies
# step.d1.d2$ens.segments
# step.d1.d2.weight$ens.accuracies
# step.d1.d2.weight$ens.segments
# 
# plot(best.acc.perpixel.d0)
# which.max(as.matrix(best.acc.perpixel.d0))
# 
# dim(est.prob.array)
# 
# install.packages("GoodmanKruskal")
# library("GoodmanKruskal")
# 
# GKtau.d0.cwr<-numeric()
# GKtau.d1.cwr<-numeric()
# GKtau.d2.cwr<-numeric()
# for(k in 1:n.x)
# {
#   GKtau.d0<-GKtau(est.classes.d0[,k], classes.temp)
#   GKtau.d0.cwr[k]<-GKtau.d0$tauxy
#   GKtau.d1<-GKtau(est.classes.d1[,k], classes.temp)
#   GKtau.d1.cwr[k]<-GKtau.d1$tauxy  
#   GKtau.d2<-GKtau(est.classes.d2[,k], classes.temp)
#   GKtau.d2.cwr[k]<-GKtau.d2$tauxy  
# }
# plot(GKtau.d0.cwr)
# plot(GKtau.d1.cwr)
# plot(GKtau.d2.cwr)
# 
# which.max(GKtau.d0.cwr)
# which(GKtaudf.test[,207]==max(GKtaudf.test[-207,207]))
# 
# system.time(GKtaudf.d0<-GKtauDataframe(est.classes.d0))
# system.time(GKtaudf.d1<-GKtauDataframe(est.classes.d1))
# system.time(GKtaudf.d2<-GKtauDataframe(est.classes.d2))
# 
# 
# system.time(GKtaudf.d0.d1.d2<-GKtauDataframe(cbind(est.classes.d0, est.classes.d1, est.classes.d2)))
# 
# diag(GKtaudf.d0)<-1
# diag(GKtaudf.d1)<-1
# diag(GKtaudf.d2)<-1
# 
# glm.1<-glm(classes.temp~est.class.probs.perpixel.d0[,1], family="binomial")
# predict(glm.1, type="response")
# 
# library("MASS")
# 
# qda.1<-qda(est.classes.d0[-1,])
# 
# library(corrplot)
# tiff("categorical_corrplot_d0.tiff")
# corrplot(GKtaudf.d0, method="color", tl.pos="n")
# dev.off()
# tiff("categorical_corrplot_d1.tiff")
# corrplot(GKtaudf.d1, method="color", tl.pos="n")
# dev.off()
# tiff("categorical_corrplot_d2.tiff")
# corrplot(GKtaudf.d2, method="color", tl.pos="n")
# dev.off()
# 
# load("Per-Pixel_Categorical.Rdata")
# 
# #load("Per-Pixel_SLE.RData")
# ### Distances per-pixel
# ### Calculate Euclidean Norm between all points at each predictor
# 
# # n.x<-length(argvals)
# # 
# # 
# # x<-lupus.fdata$data[,1]
# # 
# # plot(x)
# # dist.mat<-matrix(ncol=589, nrow=589)
# # for(j in 1:589) dist.mat[,j]<-sqrt((x[j]-x)^2)
# # out<-segment.class(dist.mat, classes.temp, 2:250, "wknn", kern.norm)
# # plot(2:250, out$accuracy.est)
# # max(out$accuracy.est)
# # which.max(out$accuracy.est)
# # out$classes.est[,which.max(out$accuracy.est)]
# # out$prob.array[,,which.max(out$accuracy.est)]
# 
# 
# 
# 
# # ### PARALLEL!!
# # 
# # use.cores<-6
# # cl.temp<-makeCluster(use.cores)
# # registerDoParallel(cl.temp)
# # 
# # predictor.matrix.temp<-lupus.fdata$data
# # n.x<-length(argvals)
# # n.samples<-length(classes.temp)
# # k.npc.grid<-2:250
# # k.npc.size<-length(k.npc.grid)
# # 
# # output.d0<-foreach(k=1:n.x, .export="segment.class", .combine=append) %dopar%
# # {
# #   #cat(k, "\n")
# #   x<-predictor.matrix.temp[,k]
# #   dist.mat<-matrix(ncol=n.samples, nrow=n.samples)
# #   for(j in 1:n.samples) dist.mat[,j]<-sqrt((x[j]-x)^2)
# #   out<-segment.class(dist.mat, classes.temp, k.npc.grid, "wknn", kern.tri)
# #   # accuracies.perpixel.d0[k,]<-out$accuracy.est
# #   # best.acc.perpixel.d0[k]<-which.max(out$accuracy.est)
# #   # est.class.probs.perpixel.d0[,k]<-out$prob.array[,2,best.acc.perpixel.d0[k]]
# #   return(list(out))
# # }
# # 
# # predictor.matrix.temp<-lupus.fdata.d1$data
# # n.x<-length(argvals)
# # n.samples<-length(classes.temp)
# # k.npc.grid<-2:250
# # k.npc.size<-length(k.npc.grid)
# # 
# # output.d1<-foreach(k=1:n.x, .export="segment.class", .combine=append) %dopar%
# # {
# #   #cat(k, "\n")
# #   x<-predictor.matrix.temp[,k]
# #   dist.mat<-matrix(ncol=n.samples, nrow=n.samples)
# #   for(j in 1:n.samples) dist.mat[,j]<-sqrt((x[j]-x)^2)
# #   out<-segment.class(dist.mat, classes.temp, k.npc.grid, "wknn", kern.tri)
# #   # accuracies.perpixel.d0[k,]<-out$accuracy.est
# #   # best.acc.perpixel.d0[k]<-which.max(out$accuracy.est)
# #   # est.class.probs.perpixel.d0[,k]<-out$prob.array[,2,best.acc.perpixel.d0[k]]
# #   return(list(out))
# # }
# # 
# # accuracies.perpixel.d0<-matrix(ncol=k.npc.size, nrow=n.x)
# # best.acc.perpixel.d0<-numeric()
# # est.class.probs.perpixel.d0<-matrix(ncol=n.x, nrow=n.samples) #col=pixels, row=patients for LOOCV, getting each patients estimated class from LOOCV for every pixel!
# # for(k in 1:n.x)
# # {
# #   best<-which.max(output.d0[[k]]$accuracy.est)
# #   accuracies.perpixel.d0[k,]<-output.d0[[k]]$accuracy.est
# #   best.acc.perpixel.d0[k]<-max(output.d0[[k]]$acc)
# #   est.class.probs.perpixel.d0[,k]<-output.d0[[k]]$prob.array[,2,best]
# # }
# # 
# # plot(best.acc.perpixel.d0)
# # 
# # accuracies.perpixel.d1<-matrix(ncol=k.npc.size, nrow=n.x)
# # best.acc.perpixel.d1<-numeric()
# # est.class.probs.perpixel.d1<-matrix(ncol=n.x, nrow=n.samples) #col=pixels, row=patients for LOOCV, getting each patients estimated class from LOOCV for every pixel!
# # for(k in 1:n.x)
# # {
# #   best<-which.max(output.d1[[k]]$accuracy.est)
# #   accuracies.perpixel.d1[k,]<-output.d1[[k]]$accuracy.est
# #   best.acc.perpixel.d1[k]<-max(output.d1[[k]]$acc)
# #   est.class.probs.perpixel.d1[,k]<-output.d1[[k]]$prob.array[,2,best]
# # }
# # 
# # plot(best.acc.perpixel.d1)
# # 
# # rm(output.d0)
# # rm(output.d1)
# # save.image("Per-Pixel_Test.RData")
# # 
# # for(k in 1:n.x)
# # {
# #   cat(k, "\n")
# #   x<-predictor.matrix.temp[,k]
# #   dist.mat<-matrix(ncol=n.samples, nrow=n.samples)
# #   for(j in 1:n.samples) dist.mat[,j]<-sqrt((x[j]-x)^2)
# #   out<-segment.class(dist.mat, classes.temp, k.npc.grid, "wknn", kern.tri)
# #   accuracies.perpixel.d0[k,]<-out$accuracy.est
# #   best.acc.perpixel.d0[k]<-which.max(out$accuracy.est)
# #   est.class.probs.perpixel.d0[,k]<-out$prob.array[,2,best.acc.perpixel.d0[k]]
# # }
# # 
# # rm(dist.mat)
# # 
# # save.image("Per-Pixel_Test.RData")
# # load("Per-Pixel_Test.RData")
# 
# 
# # mean(knn.test.d1$accuracy)
# # 
# # 
# # ### Create Probability Array and Evaluate using Stepwise Algorithms
# # temporary.seg.prob.array<-array(dim=c(589))
# # 
# # ?corrplot
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # # 
# # # 
# # # 
# # # dim(dist.mat)
# # # 
# # # plot(accuracies.perpixel.d0[1,])
# # # 
# # # 
# # # 
# # # 
# # # x<-pca.full.sle.d1$scores[,c(3)]
# # # temp.2<-metric.dist(x)
# # # 
# # # x[1,1]-x[2,1]
# # # j=1:589
# # # out<-sqrt(sum((x[1,1]-x[j,1])^2))
# # # 
# # # sqrt(sum((x[1]-x[2])^2))
# # # 
# # # dist.mat<-matrix(ncol=589, nrow=589)
# # # for(j in 1:589) dist.mat[,j]<-sqrt((x[j]-x)^2)
# # # 
# # # 
# # # out<-segment.class(dist.mat, classes.temp, 2:250, "wknn", kern.norm)
# # # plot(2:250, out$accuracy.est)
# # # max(out$accuracy.est)
# # # which.max(out$accuracy.est)
# # # out$classes.est[,which.max(out$accuracy.est)]
# # # out$prob.array[,,which.max(out$accuracy.est)]
# # # 
# # # sqrt((x[2,1]-x[j,1])^2)
# # # 
# # # 
# # # x[1,]
# # # x[2,]
# # # 
# # # 
# # # sqrt((x[1,1]-x[,2])^2)
# # # 
# # # knn()
# # # 
# # # 
# # # 
# # # x<-pca.full.sle.d2$scores[,c(1:150)]
# # # # dist.mat<-matrix(ncol=589, nrow=589)
# # # # for(j in 1:589) dist.mat[,j]<-sqrt((x[j]-x)^2)
# # # 
# # # dist.mat<-metric.dist(x)
# # # 
# # # out<-segment.class(dist.mat, classes.temp, 2:250, "wknn", kern.unif)
# # # plot(2:250, out$accuracy.est)
# # # max(out$accuracy.est)
# # # which.max(out$accuracy.est)
# # # out$classes.est[,which.max(out$accuracy.est)]
# # # out$prob.array[,,which.max(out$accuracy.est)]
# # 
