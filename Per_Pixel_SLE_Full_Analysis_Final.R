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