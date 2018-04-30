lr.fit<-function(train.x, test.x, train.classes, test.classes)
{
  N0<-length(which(test.classes==0))
  N1<-length(which(test.classes==1))
  glm.step<-glm(train.classes~., family="binomial", data=as.data.frame(t(train.x)))
  pred.step<-predict.glm(glm.step, as.data.frame(t(test.x)), type="response")
  pred.class<-rep(0,length(test.classes))
  pred.class[which(pred.step>0.5)]<-1
  pred.class[which(pred.step==0.5)]<-sample(c(0,1))
  conf.tab<-table(test.classes, pred.class)
  acc.out<-mean(test.classes==pred.class)
  
  if(length(as.numeric(colnames(conf.tab)))!=2)
  {
    missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab)))
    if(missing.obs==0) conf.tab<-cbind(c(0,0), conf.tab)
    if(missing.obs==1) conf.tab<-cbind(conf.tab, c(0,0))
  }
  
  TN.step<-conf.tab[1]
  TP.step<-conf.tab[4]
  spec.out<-TN.step/N0
  sens.out<-TP.step/N1
  
  return(list(predictions=pred.step, accuracy=acc.out, sensitivity=sens.out, specificity=spec.out))
}


lr.kcv<-function(preds, classes, folds.list)
{
  sim.full<-preds
  output.predictions<-list()
  output.accuracy<-numeric()
  output.sensitivity<-numeric()
  output.specificity<-numeric()
  
  for(k in 1:trials)
  {
    trial.temp<-foreach(j=1:folds, .combine=append, .export="lr.fit") %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      train.x<-sim.full[,folds.list[[k]]!=j]
      test.x<-sim.full[,folds.list[[k]]==j]
      
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      fit.out<-lr.fit(train.x, test.x, train.classes, test.classes)
      return(list(fit.out))
    }
    
    for(j in 1:folds)
    {
      output.predictions[[folds*(k-1)+j]]<-trial.temp[[j]]$predictions
      output.accuracy[folds*(k-1)+j]<-trial.temp[[j]]$accuracy
      output.sensitivity[folds*(k-1)+j]<-trial.temp[[j]]$sensitivity
      output.specificity[folds*(k-1)+j]<-trial.temp[[j]]$specificity 
    }
  }
  
  return(list(accuracy=output.accuracy, sensitivity=output.sensitivity, specificity=output.specificity, est.class.probs=output.predictions))
}


naive.ensembler.kcv<-function(est.class.prob.list, classes)
{
  acc.out<-numeric()
  sens.out<-numeric()
  spec.out<-numeric()
  for(k in 1:trials)
  {
    for(j in 1:folds)
    {
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      ensemble.mat<-cbind(est.class.prob.list[[1]][[folds*(k-1)+j]], est.class.prob.list[[2]][[folds*(k-1)+j]], est.class.prob.list[[3]][[folds*(k-1)+j]])
      ensemble.temp<-naive.ensembler(ensemble.mat, test.classes, N0, N1)
      acc.out[folds*(k-1)+j]<-ensemble.temp$accuracy.naive
      spec.out[folds*(k-1)+j]<-ensemble.temp$specificity.naive
      sens.out[folds*(k-1)+j]<-ensemble.temp$sensitivity.naive
    }
  }
  return(list(accuracy=acc.out, specificity=spec.out, sensitivity=sens.out))
}

weighted.ensembler.kcv<-function(accuracies, est.class.prob.list, ens.size, classes)
{
  n.sets<-length(est.class.prob.list)
  combs.to.do<-combn(n.sets, ens.size)
  outputs<-dim(combs.to.do)[2]
  
  acc.out<-matrix(nrow=folds*trials, ncol=outputs)
  sens.out<-matrix(nrow=folds*trials, ncol=outputs)
  spec.out<-matrix(nrow=folds*trials, ncol=outputs)
  for(k in 1:trials)
  {
    for(j in 1:folds)
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      ensemble.mat<-cbind(est.class.prob.list[[1]][[folds*(k-1)+j]], est.class.prob.list[[2]][[folds*(k-1)+j]], est.class.prob.list[[3]][[folds*(k-1)+j]])
      ensemble.temp<-weighted.ensembler(accuracies[folds*(k-1)+j,],ensemble.mat, ens.size, test.classes, N0, N1)
      acc.out[folds*(k-1)+j,]<-ensemble.temp$accuracy.ens
      spec.out[folds*(k-1)+j,]<-ensemble.temp$specificity.ens
      sens.out[folds*(k-1)+j,]<-ensemble.temp$sensitivity.ens
    }
  }
  return(list(accuracy=acc.out, specificity=spec.out, sensitivity=sens.out))
}

lasso.fit<-function(train.x, test.x, train.classes, test.classes)
{
  ### LASSO Regression ###
  N0<-length(which(test.classes==0))
  N1<-length(which(test.classes==1))
  
  cv.step<-cv.glmnet(x=t(train.x), y=train.classes, family="binomial", alpha=1)
  glm.step<-glmnet(x=t(train.x), y=train.classes, family="binomial", alpha=1, lambda=cv.step$lambda.min)
  pred.step<-predict(glm.step, t(test.x), type="response", s=cv.step$lambda.min)
  pred.class<-rep(0,length(test.classes))
  pred.class[which(pred.step>0.5)]<-1
  conf.tab<-table(test.classes, pred.class)
  acc.out<-mean(test.classes==pred.class)
  coefs.out<-as.vector(coef(glm.step))
  
  if(length(as.numeric(colnames(conf.tab)))!=2)
  {
    missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab)))
    if(missing.obs==0) conf.tab<-cbind(c(0,0), conf.tab)
    if(missing.obs==1) conf.tab<-cbind(conf.tab, c(0,0))
  }
  
  TN.step<-conf.tab[1]
  TP.step<-conf.tab[4]
  spec.out<-TN.step/N0
  sens.out<-TP.step/N1
  
  return(list(predictions=pred.step, accuracy=acc.out, sensitivity=sens.out, specificity=spec.out))
}

lasso.kcv<-function(preds, classes, folds.list)
{
  sim.full<-preds
  output.predictions<-list()
  output.accuracy<-numeric()
  output.sensitivity<-numeric()
  output.specificity<-numeric()
  
  for(k in 1:trials)
  {
    trial.temp<-foreach(j=1:folds, .combine=append, .export=c("lasso.fit"), .packages="glmnet") %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      train.x<-sim.full[,folds.list[[k]]!=j]
      test.x<-sim.full[,folds.list[[k]]==j]
      
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      fit.out<-lasso.fit(train.x, test.x, train.classes, test.classes)
      return(list(fit.out))
    }
    
    for(j in 1:folds)
    {
      output.predictions[[folds*(k-1)+j]]<-trial.temp[[j]]$predictions
      output.accuracy[folds*(k-1)+j]<-trial.temp[[j]]$accuracy
      output.sensitivity[folds*(k-1)+j]<-trial.temp[[j]]$sensitivity
      output.specificity[folds*(k-1)+j]<-trial.temp[[j]]$specificity 
    }
  }
  
  return(list(accuracy=output.accuracy, sensitivity=output.sensitivity, specificity=output.specificity, est.class.probs=output.predictions))
}

ridge.fit<-function(train.x, test.x, train.classes, test.classes)
{
  ### ridge Regression ###
  N0<-length(which(test.classes==0))
  N1<-length(which(test.classes==1))
  
  cv.step<-cv.glmnet(x=t(train.x), y=train.classes, family="binomial", alpha=0)
  glm.step<-glmnet(x=t(train.x), y=train.classes, family="binomial", alpha=0, lambda=cv.step$lambda.min)
  pred.step<-predict(glm.step, t(test.x), type="response", s=cv.step$lambda.min)
  pred.class<-rep(0,length(test.classes))
  pred.class[which(pred.step>0.5)]<-1
  conf.tab<-table(test.classes, pred.class)
  acc.out<-mean(test.classes==pred.class)
  coefs.out<-as.vector(coef(glm.step))
  
  if(length(as.numeric(colnames(conf.tab)))!=2)
  {
    missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab)))
    if(missing.obs==0) conf.tab<-cbind(c(0,0), conf.tab)
    if(missing.obs==1) conf.tab<-cbind(conf.tab, c(0,0))
  }
  
  TN.step<-conf.tab[1]
  TP.step<-conf.tab[4]
  spec.out<-TN.step/N0
  sens.out<-TP.step/N1
  
  return(list(predictions=pred.step, accuracy=acc.out, sensitivity=sens.out, specificity=spec.out))
}

ridge.kcv<-function(preds, classes, folds.list)
{
  sim.full<-preds
  output.predictions<-list()
  output.accuracy<-numeric()
  output.sensitivity<-numeric()
  output.specificity<-numeric()
  
  for(k in 1:trials)
  {
    trial.temp<-foreach(j=1:folds, .combine=append, .export=c("ridge.fit"), .packages="glmnet") %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      train.x<-sim.full[,folds.list[[k]]!=j]
      test.x<-sim.full[,folds.list[[k]]==j]
      
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      fit.out<-ridge.fit(train.x, test.x, train.classes, test.classes)
      return(list(fit.out))
    }
    
    for(j in 1:folds)
    {
      output.predictions[[folds*(k-1)+j]]<-trial.temp[[j]]$predictions
      output.accuracy[folds*(k-1)+j]<-trial.temp[[j]]$accuracy
      output.sensitivity[folds*(k-1)+j]<-trial.temp[[j]]$sensitivity
      output.specificity[folds*(k-1)+j]<-trial.temp[[j]]$specificity 
    }
  }
  
  return(list(accuracy=output.accuracy, sensitivity=output.sensitivity, specificity=output.specificity, est.class.probs=output.predictions))
}

enet.fit<-function(train.x, test.x, train.classes, test.classes)
{
  ### enet Regression ###
  N0<-length(which(test.classes==0))
  N1<-length(which(test.classes==1))
  
  cv.step<-cv.glmnet(x=t(train.x), y=train.classes, family="binomial", alpha=0.5)
  glm.step<-glmnet(x=t(train.x), y=train.classes, family="binomial", alpha=0.5, lambda=cv.step$lambda.min)
  pred.step<-predict(glm.step, t(test.x), type="response", s=cv.step$lambda.min)
  pred.class<-rep(0,length(test.classes))
  pred.class[which(pred.step>0.5)]<-1
  conf.tab<-table(test.classes, pred.class)
  acc.out<-mean(test.classes==pred.class)
  coefs.out<-as.vector(coef(glm.step))
  
  if(length(as.numeric(colnames(conf.tab)))!=2)
  {
    missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab)))
    if(missing.obs==0) conf.tab<-cbind(c(0,0), conf.tab)
    if(missing.obs==1) conf.tab<-cbind(conf.tab, c(0,0))
  }
  
  TN.step<-conf.tab[1]
  TP.step<-conf.tab[4]
  spec.out<-TN.step/N0
  sens.out<-TP.step/N1
  
  return(list(predictions=pred.step, accuracy=acc.out, sensitivity=sens.out, specificity=spec.out))
}

enet.kcv<-function(preds, classes, folds.list)
{
  sim.full<-preds
  output.predictions<-list()
  output.accuracy<-numeric()
  output.sensitivity<-numeric()
  output.specificity<-numeric()
  
  for(k in 1:trials)
  {
    trial.temp<-foreach(j=1:folds, .combine=append, .export=c("enet.fit"), .packages="glmnet") %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      train.x<-sim.full[,folds.list[[k]]!=j]
      test.x<-sim.full[,folds.list[[k]]==j]
      
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      fit.out<-enet.fit(train.x, test.x, train.classes, test.classes)
      return(list(fit.out))
    }
    
    for(j in 1:folds)
    {
      output.predictions[[folds*(k-1)+j]]<-trial.temp[[j]]$predictions
      output.accuracy[folds*(k-1)+j]<-trial.temp[[j]]$accuracy
      output.sensitivity[folds*(k-1)+j]<-trial.temp[[j]]$sensitivity
      output.specificity[folds*(k-1)+j]<-trial.temp[[j]]$specificity 
    }
  }
  
  return(list(accuracy=output.accuracy, sensitivity=output.sensitivity, specificity=output.specificity, est.class.probs=output.predictions))
}

LDA.fit<-function(train.x, test.x, train.classes, test.classes)
{
  N0<-length(which(test.classes==0))
  N1<-length(which(test.classes==1))
  n.x<-ncol(t(test.x))
  train.df<-as.data.frame(cbind(train.classes, t(train.x)))
  test.df<-as.data.frame(cbind(test.classes, t(test.x)))
  
  ### LDA ###
  lda.fit.temp<-lda(train.classes~., data=train.df)
  lda.pred.temp<- predict(lda.fit.temp, test.df[,c(2:(n.x+1))])
  acc.out<-mean(lda.pred.temp$class==test.classes)
  conf.tab<-table(test.classes, lda.pred.temp$class)
  
  if(length(as.numeric(colnames(conf.tab)))!=2)
  {
    missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab)))
    if(missing.obs==0) conf.tab<-cbind(c(0,0), conf.tab)
    if(missing.obs==1) conf.tab<-cbind(conf.tab, c(0,0))
  }
  
  TN.step<-conf.tab[1]
  TP.step<-conf.tab[4]
  spec.out<-TN.step/N0
  sens.out<-TP.step/N1
  
  return(list(predictions=lda.pred.temp$posterior[,2], accuracy=acc.out, sensitivity=sens.out, specificity=spec.out))
}

LDA.kcv<-function(preds, classes, folds.list)
{
  sim.full<-preds
  output.predictions<-list()
  output.accuracy<-numeric()
  output.sensitivity<-numeric()
  output.specificity<-numeric()
  
  for(k in 1:trials)
  {
    trial.temp<-foreach(j=1:folds, .combine=append, .export=c("LDA.fit"), .packages="MASS") %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      train.x<-sim.full[,folds.list[[k]]!=j]
      test.x<-sim.full[,folds.list[[k]]==j]
      
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      fit.out<-LDA.fit(train.x, test.x, train.classes, test.classes)
      return(list(fit.out))
    }
    
    for(j in 1:folds)
    {
      output.predictions[[folds*(k-1)+j]]<-trial.temp[[j]]$predictions
      output.accuracy[folds*(k-1)+j]<-trial.temp[[j]]$accuracy
      output.sensitivity[folds*(k-1)+j]<-trial.temp[[j]]$sensitivity
      output.specificity[folds*(k-1)+j]<-trial.temp[[j]]$specificity 
    }
  }
  
  return(list(accuracy=output.accuracy, sensitivity=output.sensitivity, specificity=output.specificity, est.class.probs=output.predictions))
}

QDA.fit<-function(train.x, test.x, train.classes, test.classes)
{
  N0<-length(which(test.classes==0))
  N1<-length(which(test.classes==1))
  n.x<-ncol(t(test.x))
  train.df<-as.data.frame(cbind(train.classes, t(train.x)))
  test.df<-as.data.frame(cbind(test.classes, t(test.x)))
  
  ### QDA ###
  QDA.fit.temp<-qda(train.classes~., data=train.df)
  QDA.pred.temp<- predict(QDA.fit.temp, test.df[,c(2:(n.x+1))])
  acc.out<-mean(QDA.pred.temp$class==test.classes)
  conf.tab<-table(test.classes, QDA.pred.temp$class)
  
  if(length(as.numeric(colnames(conf.tab)))!=2)
  {
    missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab)))
    if(missing.obs==0) conf.tab<-cbind(c(0,0), conf.tab)
    if(missing.obs==1) conf.tab<-cbind(conf.tab, c(0,0))
  }
  
  TN.step<-conf.tab[1]
  TP.step<-conf.tab[4]
  spec.out<-TN.step/N0
  sens.out<-TP.step/N1
  
  return(list(predictions=QDA.pred.temp$posterior[,2], accuracy=acc.out, sensitivity=sens.out, specificity=spec.out))
}

QDA.kcv<-function(preds, classes, folds.list)
{
  sim.full<-preds
  output.predictions<-list()
  output.accuracy<-numeric()
  output.sensitivity<-numeric()
  output.specificity<-numeric()
  
  for(k in 1:trials)
  {
    trial.temp<-foreach(j=1:folds, .combine=append, .export=c("QDA.fit"), .packages="MASS") %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      train.x<-sim.full[,folds.list[[k]]!=j]
      test.x<-sim.full[,folds.list[[k]]==j]
      
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      fit.out<-QDA.fit(train.x, test.x, train.classes, test.classes)
      return(list(fit.out))
    }
    
    for(j in 1:folds)
    {
      output.predictions[[folds*(k-1)+j]]<-trial.temp[[j]]$predictions
      output.accuracy[folds*(k-1)+j]<-trial.temp[[j]]$accuracy
      output.sensitivity[folds*(k-1)+j]<-trial.temp[[j]]$sensitivity
      output.specificity[folds*(k-1)+j]<-trial.temp[[j]]$specificity 
    }
  }
  
  return(list(accuracy=output.accuracy, sensitivity=output.sensitivity, specificity=output.specificity, est.class.probs=output.predictions))
}

knn.fit<-function(train.x, test.x, train.classes, test.classes, k.npc)
{
  N0<-length(which(test.classes==0))
  N1<-length(which(test.classes==1))
  n.x<-ncol(t(test.x))
  train.df<-as.data.frame(cbind(train.classes, t(train.x)))
  test.df<-as.data.frame(cbind(test.classes, t(test.x)))
  
  ### QDA ###
  knn.test<-knn(t(train.x), t(test.x), train.classes, k=k.npc, prob = TRUE)
  probs.temp<-attributes(knn.test)$prob
  probs.temp[which(knn.test==0)]<-1-probs.temp[which(knn.test==0)]
  conf.tab<-table(test.classes, knn.test)
  acc.out<-mean(test.classes==knn.test)
  
  if(length(as.numeric(colnames(conf.tab)))!=2)
  {
    missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab)))
    if(missing.obs==0) conf.tab<-cbind(c(0,0), conf.tab)
    if(missing.obs==1) conf.tab<-cbind(conf.tab, c(0,0))
  }
  
  TN.step<-conf.tab[1]
  TP.step<-conf.tab[4]
  spec.out<-TN.step/N0
  sens.out<-TP.step/N1
  
  return(list(predictions=probs.temp, accuracy=acc.out, sensitivity=sens.out, specificity=spec.out))
}

knn.kcv<-function(preds, classes, folds.list, k.npc.grid)
{
  sim.full<-preds
  output.predictions<-list(list())
  output.accuracy<-matrix(ncol=length(k.npc.grid), nrow=folds*trials)
  output.sensitivity<-matrix(ncol=length(k.npc.grid), nrow=folds*trials)
  output.specificity<-matrix(ncol=length(k.npc.grid), nrow=folds*trials)
  # browser()
  for(k in 1:trials)
  {
    trial.temp<-foreach(j=1:folds, .combine=append, .export=c("knn.fit"), .packages="class") %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      train.x<-sim.full[,folds.list[[k]]!=j]
      test.x<-sim.full[,folds.list[[k]]==j]
      
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      
      fit.list.k<-list()
      for(m in 1:length(k.npc.grid))
      {
        k.npc<-k.npc.grid[m]
        fit.list.k[[m]]<-knn.fit(train.x, test.x, train.classes, test.classes, k.npc)
      }
      
      return(list(fit.list.k))
    }
    
    for(j in 1:folds)
    {
      output.predictions[[folds*(k-1)+j]]<-list()
      for(m in 1:length(k.npc.grid))
      {
        output.predictions[[folds*(k-1)+j]][[m]]<-trial.temp[[j]][[m]]$predictions
        output.accuracy[folds*(k-1)+j,m]<-trial.temp[[j]][[m]]$accuracy
        output.sensitivity[folds*(k-1)+j,m]<-trial.temp[[j]][[m]]$sensitivity
        output.specificity[folds*(k-1)+j,m]<-trial.temp[[j]][[m]]$specificity   
      }
    }
  }
  
  return(list(accuracy=output.accuracy, sensitivity=output.sensitivity, specificity=output.specificity, est.class.probs=output.predictions))
}



