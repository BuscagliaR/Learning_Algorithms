## Dr. Robert Buscaglia
## rbuscagl@asu.edu
## April 2018

source("ESFuNC_Functions_March_2018.R")
source("Ensemble_Functions.R")
source("Supervised_Learning_Algorithms.R")

#############################
### lupuskernelnorm dataset       ###
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

### REDUCED BASIS NO SMOOTHING ###  GCV in MANUSCRIPT
lupus.fd.GCV<-fdata2fd(lupus.fdata, nbasis=150, lambda=0)
lupus.fdata.GCV<-fdata(t(eval.fd(argvals, lupus.fd.GCV)), argvals=argvals)

lupus.fd.GCV.d1<-fdata2fd(lupus.fdata, nbasis=150, lambda=0, nderiv=1)
lupus.fdata.GCV.d1<-fdata(t(eval.fd(argvals, lupus.fd.GCV.d1)), argvals=argvals)

lupus.fd.GCV.d2<-fdata2fd(lupus.fdata, nbasis=150, lambda=0, nderiv=2)
lupus.fdata.GCV.d2<-fdata(t(eval.fd(argvals, lupus.fd.GCV.d2)), argvals=argvals)

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


### FPCA ANALYSIS : SCORES ARE 

pca.full.sle.d0<-pca.fd(lupus.fd, nharm=451)
pca.full.sle.d1<-pca.fd(lupus.fd.d1, nharm=451)
pca.full.sle.d2<-pca.fd(lupus.fd.d2, nharm=451)


### QDA ###

folds=10
trials=2
pcs.to.test<-seq(2, 260, 2)

use.cores<-6
cl.temp<-makeCluster(use.cores)
registerDoParallel(cl.temp)

predictions.pc.QDA.d0<-list()
accuracy.pc.QDA.d0<-matrix(ncol=length(pcs.to.test), nrow=folds*trials)
sensitivity.pc.QDA.d0<-matrix(ncol=length(pcs.to.test), nrow=folds*trials)
specificity.pc.QDA.d0<-matrix(ncol=length(pcs.to.test), nrow=folds*trials)

for(k in 1:length(pcs.to.test))
{
  pcs<-pcs.to.test[k]
  cat(pcs, "\n")
  preds<-t(pca.full.sle.d0$scores[,1:pcs])
  sle.pca.QDA.temp.d0<-QDA.kcv(preds, classes.temp, folds.list)
  predictions.pc.QDA.d0[[k]]<-sle.pca.QDA.temp.d0$est.class.probs
  accuracy.pc.QDA.d0[,k]<-sle.pca.QDA.temp.d0$accuracy
  sensitivity.pc.QDA.d0[,k]<-sle.pca.QDA.temp.d0$sensitivity
  specificity.pc.QDA.d0[,k]<-sle.pca.QDA.temp.d0$specificity
}

save.image("PCA_LR_QDA_SLE.RData")

predictions.pc.QDA.d1<-list()
accuracy.pc.QDA.d1<-matrix(ncol=length(pcs.to.test), nrow=folds*trials)
sensitivity.pc.QDA.d1<-matrix(ncol=length(pcs.to.test), nrow=folds*trials)
specificity.pc.QDA.d1<-matrix(ncol=length(pcs.to.test), nrow=folds*trials)

for(k in 1:length(pcs.to.test))
{
  pcs<-pcs.to.test[k]
  cat(pcs, "\n")
  preds<-t(pca.full.sle.d1$scores[,1:pcs])
  sle.pca.QDA.temp.d1<-QDA.kcv(preds, classes.temp, folds.list)
  predictions.pc.QDA.d1[[k]]<-sle.pca.QDA.temp.d1$est.class.probs
  accuracy.pc.QDA.d1[,k]<-sle.pca.QDA.temp.d1$accuracy
  sensitivity.pc.QDA.d1[,k]<-sle.pca.QDA.temp.d1$sensitivity
  specificity.pc.QDA.d1[,k]<-sle.pca.QDA.temp.d1$specificity
}

save.image("PCA_LR_QDA_SLE.RData")

predictions.pc.QDA.d2<-list()
accuracy.pc.QDA.d2<-matrix(ncol=length(pcs.to.test), nrow=folds*trials)
sensitivity.pc.QDA.d2<-matrix(ncol=length(pcs.to.test), nrow=folds*trials)
specificity.pc.QDA.d2<-matrix(ncol=length(pcs.to.test), nrow=folds*trials)

for(k in 1:length(pcs.to.test))
{
  pcs<-pcs.to.test[k]
  cat(pcs, "\n")
  preds<-t(pca.full.sle.d2$scores[,1:pcs])
  sle.pca.QDA.temp.d2<-QDA.kcv(preds, classes.temp, folds.list)
  predictions.pc.QDA.d2[[k]]<-sle.pca.QDA.temp.d2$est.class.probs
  accuracy.pc.QDA.d2[,k]<-sle.pca.QDA.temp.d2$accuracy
  sensitivity.pc.QDA.d2[,k]<-sle.pca.QDA.temp.d2$sensitivity
  specificity.pc.QDA.d2[,k]<-sle.pca.QDA.temp.d2$specificity
}

save.image("PCA_LR_QDA_SLE.RData")

boxplot(accuracy.pc.QDA.d0)
lines(colMeans(accuracy.pc.QDA.d0), col="red", lwd=3)
max(colMeans(accuracy.pc.QDA.d0))
pc.QDA.d0.best<-which.max(colMeans(accuracy.pc.QDA.d0))
est.class.prob.d0<-predictions.pc.QDA.d0[[pc.QDA.d0.best]]
accuracy.d0<-accuracy.pc.QDA.d0[,pc.QDA.d0.best]

boxplot(accuracy.pc.QDA.d1)
lines(colMeans(accuracy.pc.QDA.d1), col="red", lwd=3)
max(colMeans(accuracy.pc.QDA.d1))
pc.QDA.d1.best<-which.max(colMeans(accuracy.pc.QDA.d1))
est.class.prob.d1<-predictions.pc.QDA.d1[[pc.QDA.d1.best]]
accuracy.d1<-accuracy.pc.QDA.d1[,pc.QDA.d1.best]

boxplot(accuracy.pc.QDA.d2)
lines(colMeans(accuracy.pc.QDA.d2), col="red", lwd=3)
max(colMeans(accuracy.pc.QDA.d2))
pc.QDA.d2.best<-which.max(colMeans(accuracy.pc.QDA.d2))
est.class.prob.d2<-predictions.pc.QDA.d2[[pc.QDA.d2.best]]
accuracy.d2<-accuracy.pc.QDA.d2[,pc.QDA.d2.best]

stopCluster(cl.temp)

save.image("PCA_LR_QDA_SLE.RData")

accuracies.mat<-cbind(accuracy.d0, accuracy.d1, accuracy.d2)
est.class.prob.list<-list(est.class.prob.d0, est.class.prob.d1, est.class.prob.d2)
classes<-classes.temp
length(est.class.prob.list)

naive.ens<-naive.ensembler.kcv(est.class.prob.list, classes.temp)
mean(naive.ens$accuracy)
boxplot(naive.ens$accuracy)

acc.weighted.ens.2<-weighted.ensembler.kcv(accuracies.mat, est.class.prob.list, 2, classes.temp)
colMeans(acc.weighted.ens.2$accuracy)
boxplot(acc.weighted.ens.2$accuracy)

acc.weighted.ens.3<-weighted.ensembler.kcv(accuracies.mat, est.class.prob.list, 3, classes.temp)
colMeans(acc.weighted.ens.3$accuracy)
boxplot(acc.weighted.ens.3$accuracy)

equal.mat<-cbind(rep(1,folds*trials), rep(1,folds*trials), rep(1,folds*trials))
equal.weighted.ens.2<-weighted.ensembler.kcv(equal.mat, est.class.prob.list, 2, classes.temp)
colMeans(equal.weighted.ens.2$accuracy)
boxplot(equal.weighted.ens.2$accuracy)

equal.weighted.ens.3<-weighted.ensembler.kcv(equal.mat, est.class.prob.list, 3, classes.temp)
colMeans(equal.weighted.ens.3$accuracy)
boxplot(equal.weighted.ens.3$accuracy)

save.image("PCA_LR_QDA_SLE.RData")
