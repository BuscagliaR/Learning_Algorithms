########## ENSEMBLERS ###############
### USED IN DISSERTATION
### CREATES ENSEMBLE FOR DIFFERENT CLASSIFIERS

### ENSEMBLE ###

naive.ensembler<-function(predictions.matrix, test.classes, N0, N1)
{
  n.sets<-dim(predictions.matrix)[2]
  n.test<-dim(predictions.matrix)[1]
  
  classes.naive<-matrix(0, ncol=n.sets, nrow=n.test)
  for(m in 1:n.sets)
  {
    classes.naive[which(predictions.matrix[,m]>0.5),m]<-1
    classes.naive[which(predictions.matrix[,m]==0.5),m]<-sample(c(0,1),1)
  }
  classes.naive
  pred.class.niave.ens<-rep(0, n.test)
  pred.class.niave.ens[which(rowMeans(classes.naive)>0.5)]<-1
  pred.class.niave.ens[which(rowMeans(classes.naive)==0.5)]<-sample(c(0,1),1)
  conf.tab.naive<-table(test.classes,pred.class.niave.ens)
  accuracy.naive<-mean(test.classes==pred.class.niave.ens)
  
  if(length(as.numeric(colnames(conf.tab.naive)))!=2)
  {
    missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab.naive)))
    if(missing.obs==0) conf.tab.naive<-cbind(c(0,0), conf.tab.naive)
    if(missing.obs==1) conf.tab.naive<-cbind(conf.tab.naive, c(0,0))
  }
  
  TN.naive<-conf.tab.naive[1]
  TP.naive<-conf.tab.naive[4]
  specificity.naive<-TN.naive/N0
  sensitivity.naive<-TP.naive/N1
  
  return(list(accuracy.naive=accuracy.naive, specificity.naive=specificity.naive, sensitivity.naive=sensitivity.naive))
}


weighted.ensembler<-function(accuracy.vec, predictions.mat, ens.size, test.classes, N0, N1)
{
  m<-ens.size
  n.sets<-dim(predictions.mat)[2]
  n.test<-dim(predictions.mat)[1]
  combs.to.do<-combn(n.sets, m)
  
  accuracy.ens<-numeric()
  specificity.ens<-numeric()
  sensitivity.ens<-numeric()
  
  for(p in 1:dim(combs.to.do)[2])
  {
    ens.probs<-rowSums(t(accuracy.vec[combs.to.do[,p]]*t(predictions.mat[,combs.to.do[,p]])))/(sum(accuracy.vec[combs.to.do[,p]]))
    pred.class.ens<-rep(0,n.test)
    pred.class.ens[which(ens.probs>0.5)]<-1
    pred.class.ens[which(ens.probs==0.5)]<-sample(c(0,1),1)
    conf.tab.ens<-table(test.classes, pred.class.ens)
    accuracy.ens[p]<-mean(test.classes==pred.class.ens)
    
    if(length(as.numeric(colnames(conf.tab.ens)))!=2)
    {
      missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab.ens)))
      if(missing.obs==0) conf.tab.ens<-cbind(c(0,0), conf.tab.ens)
      if(missing.obs==1) conf.tab.ens<-cbind(conf.tab.ens, c(0,0))
    }
    
    TN.ens<-conf.tab.ens[1]
    TP.ens<-conf.tab.ens[4]
    specificity.ens[p]<-TN.ens/N0
    sensitivity.ens[p]<-TP.ens/N1
  }
  
  return(list(accuracy.ens=accuracy.ens, specificity.ens=specificity.ens, sensitivity.ens=sensitivity.ens))
}

equal.weighted.ensembler<-function(predictions.mat, ens.size)
{
  m=ens.size
  combs.to.do<-combn(n.sets, m)
  
  accuracy.ens<-numeric()
  specificity.ens<-numeric()
  sensitivity.ens<-numeric()
  
  for(p in 1:dim(combs.to.do)[2])
  {
    ens.probs<-rowMeans(predictions.mat[,combs.to.do[,p]])
    pred.class.ens<-rep(0,n.test)
    pred.class.ens[which(ens.probs>0.5)]<-1
    pred.class.ens[which(ens.probs==0.5)]<-sample(c(0,1),1)
    conf.tab.ens<-table(test.classes, pred.class.ens)
    accuracy.ens[p]<-mean(test.classes==pred.class.ens)
    
    if(length(as.numeric(colnames(conf.tab.ens)))!=2)
    {
      missing.obs<-setdiff(c(0,1), as.numeric(colnames(conf.tab.ens)))
      if(missing.obs==0) conf.tab.ens<-cbind(c(0,0), conf.tab.ens)
      if(missing.obs==1) conf.tab.ens<-cbind(conf.tab.ens, c(0,0))
    }
    
    TN.ens<-conf.tab.ens[1]
    TP.ens<-conf.tab.ens[4]
    specificity.ens[p]<-TN.ens/N0
    sensitivity.ens[p]<-TP.ens/N1
  }
  
  return(list(accuracy.ens=accuracy.ens, specificity.ens=specificity.ens, sensitivity.ens=sensitivity.ens))
}

# use.cores=6

ensemble.lr.sim<-function(preds.list, classes, n.x, folds.list)
{
  ### Begin Function
  n.sets<-length(preds.list)
  accuracy.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  sensitivity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  specificity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  
  colnames(accuracy.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(sensitivity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(specificity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  
  # use.cores<-6
  # cl.temp<-makeCluster(use.cores)
  # registerDoParallel(cl.temp)
  
  for(k in 1:trials)
  {
    cat("Trial ", k, "\n")
    trial.temp<-foreach(j=1:folds, .combine=append, .export=c("naive.ensembler", "weighted.ensembler", "equal.weighted.ensembler")) %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      ### OUTPUTS FROM FOREACH
      #predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.all<-numeric()
      specificity.all<-numeric()
      sensitivity.all<-numeric()
      
      ### OUTPUTS FROM SOLO ANALYSIS
      predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.solo<-numeric()
      specificity.solo<-numeric()
      sensitivity.solo<-numeric()
      
      ### ACCURACIES AND PREDICTION PROBABILITIES FOR EACH SET OF PREDICTORS
      for(m in 1:n.sets)
      {
        sim.full<-preds.list[[m]]
        
        train.x<-sim.full[,folds.list[[k]]!=j]
        test.x<-sim.full[,folds.list[[k]]==j]
        
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
        
        predictions.solo[,m]<-pred.step
        accuracy.solo[m]<-acc.out
        specificity.solo[m]<-spec.out
        sensitivity.solo[m]<-sens.out
      }
      
      #STORE SINGLE CLASSIFIER RESULTS
      accuracy.all[1:n.sets]<-accuracy.solo
      specificity.all[1:n.sets]<-specificity.solo
      sensitivity.all[1:n.sets]<-sensitivity.solo
      
      ### RETURN AND STORE NAIVE ENSEMBLE RESULTS
      naive.ens.out<-naive.ensembler(predictions.solo)
      accuracy.all[n.sets+1]<-naive.ens.out$accuracy.naive
      specificity.all[n.sets+1]<-naive.ens.out$specificity.naive
      sensitivity.all[n.sets+1]<-naive.ens.out$sensitivity.naive
      
      ### RETURN ACCURACY WEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=2)
      accuracy.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+5)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+5)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+5)]<-ens.temp.result$sensitivity.ens
      
      ### RETURN ACCURACY UNWEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=2)
      accuracy.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+9)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+9)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+9)]<-ens.temp.result$sensitivity.ens
      
      output<-list(accuracy.all=accuracy.all, specificity.all=specificity.all, sensitivity.all=sensitivity.all)
      return(list(output))
    }
    
    for(j in 1:folds)
    {
      accuracy.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$accuracy.all
      specificity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$specificity.all
      sensitivity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$sensitivity.all
    }
  }
  
  # stopCluster(cl.temp)
  
  return(list(accuracy=accuracy.ensemble.matrix, specificity=specificity.ensemble.matrix, sensitivity=sensitivity.ensemble.matrix))
  
}

ensemble.lasso.sim<-function(preds.list, classes, n.x, folds.list)
{
  ### Begin Function
  n.sets<-length(preds.list)
  accuracy.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  sensitivity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  specificity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  
  colnames(accuracy.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(sensitivity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(specificity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  
  # use.cores<-6
  # cl.temp<-makeCluster(use.cores)
  # registerDoParallel(cl.temp)
  
  for(k in 1:trials)
  {
    cat("Trial ", k, "\n")
    trial.temp<-foreach(j=1:folds, .combine=append, .packages=c("glmnet"), .export=c("naive.ensembler", "weighted.ensembler", "equal.weighted.ensembler")) %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      ### OUTPUTS FROM FOREACH
      #predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.all<-numeric()
      specificity.all<-numeric()
      sensitivity.all<-numeric()
      
      ### OUTPUTS FROM SOLO ANALYSIS
      predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.solo<-numeric()
      specificity.solo<-numeric()
      sensitivity.solo<-numeric()
      
      ### ACCURACIES AND PREDICTION PROBABILITIES FOR EACH SET OF PREDICTORS
      for(m in 1:n.sets)
      {
        sim.full<-preds.list[[m]]
        
        train.x<-sim.full[,folds.list[[k]]!=j]
        test.x<-sim.full[,folds.list[[k]]==j]
        
        ### LASSO Regression ###
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
        
        predictions.solo[,m]<-pred.step
        accuracy.solo[m]<-acc.out
        specificity.solo[m]<-spec.out
        sensitivity.solo[m]<-sens.out
      }
      
      #STORE SINGLE CLASSIFIER RESULTS
      accuracy.all[1:n.sets]<-accuracy.solo
      specificity.all[1:n.sets]<-specificity.solo
      sensitivity.all[1:n.sets]<-sensitivity.solo
      
      ### RETURN AND STORE NAIVE ENSEMBLE RESULTS
      naive.ens.out<-naive.ensembler(predictions.solo)
      accuracy.all[n.sets+1]<-naive.ens.out$accuracy.naive
      specificity.all[n.sets+1]<-naive.ens.out$specificity.naive
      sensitivity.all[n.sets+1]<-naive.ens.out$sensitivity.naive
      
      ### RETURN ACCURACY WEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=2)
      accuracy.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+5)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+5)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+5)]<-ens.temp.result$sensitivity.ens
      
      ### RETURN ACCURACY UNWEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=2)
      accuracy.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+9)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+9)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+9)]<-ens.temp.result$sensitivity.ens
      
      output<-list(accuracy.all=accuracy.all, specificity.all=specificity.all, sensitivity.all=sensitivity.all)
      return(list(output))
    }
    
    for(j in 1:folds)
    {
      accuracy.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$accuracy.all
      specificity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$specificity.all
      sensitivity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$sensitivity.all
    }
  }
  # stopCluster(cl.temp)
  return(list(accuracy=accuracy.ensemble.matrix, specificity=specificity.ensemble.matrix, sensitivity=sensitivity.ensemble.matrix))
  
}

ensemble.ridge.sim<-function(preds.list, classes, n.x, folds.list)
{
  ### Begin Function
  n.sets<-length(preds.list)
  accuracy.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  sensitivity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  specificity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  
  colnames(accuracy.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(sensitivity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(specificity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  
  # use.cores<-6
  # cl.temp<-makeCluster(use.cores)
  # registerDoParallel(cl.temp)
  
  for(k in 1:trials)
  {
    cat("Trial ", k, "\n")
    trial.temp<-foreach(j=1:folds, .combine=append, .packages=c("glmnet"), .export=c("naive.ensembler", "weighted.ensembler", "equal.weighted.ensembler")) %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      ### OUTPUTS FROM FOREACH
      #predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.all<-numeric()
      specificity.all<-numeric()
      sensitivity.all<-numeric()
      
      ### OUTPUTS FROM SOLO ANALYSIS
      predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.solo<-numeric()
      specificity.solo<-numeric()
      sensitivity.solo<-numeric()
      
      ### ACCURACIES AND PREDICTION PROBABILITIES FOR EACH SET OF PREDICTORS
      for(m in 1:n.sets)
      {
        sim.full<-preds.list[[m]]
        
        train.x<-sim.full[,folds.list[[k]]!=j]
        test.x<-sim.full[,folds.list[[k]]==j]
        
        ### RIDGE Regression ###
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
        
        predictions.solo[,m]<-pred.step
        accuracy.solo[m]<-acc.out
        specificity.solo[m]<-spec.out
        sensitivity.solo[m]<-sens.out
      }
      
      #STORE SINGLE CLASSIFIER RESULTS
      accuracy.all[1:n.sets]<-accuracy.solo
      specificity.all[1:n.sets]<-specificity.solo
      sensitivity.all[1:n.sets]<-sensitivity.solo
      
      ### RETURN AND STORE NAIVE ENSEMBLE RESULTS
      naive.ens.out<-naive.ensembler(predictions.solo)
      accuracy.all[n.sets+1]<-naive.ens.out$accuracy.naive
      specificity.all[n.sets+1]<-naive.ens.out$specificity.naive
      sensitivity.all[n.sets+1]<-naive.ens.out$sensitivity.naive
      
      ### RETURN ACCURACY WEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=2)
      accuracy.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+5)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+5)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+5)]<-ens.temp.result$sensitivity.ens
      
      ### RETURN ACCURACY UNWEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=2)
      accuracy.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+9)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+9)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+9)]<-ens.temp.result$sensitivity.ens
      
      output<-list(accuracy.all=accuracy.all, specificity.all=specificity.all, sensitivity.all=sensitivity.all)
      return(list(output))
    }
    
    for(j in 1:folds)
    {
      accuracy.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$accuracy.all
      specificity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$specificity.all
      sensitivity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$sensitivity.all
    }
  }
  # stopCluster(cl.temp)
  return(list(accuracy=accuracy.ensemble.matrix, specificity=specificity.ensemble.matrix, sensitivity=sensitivity.ensemble.matrix))
  
}

ensemble.enet.sim<-function(preds.list, classes, n.x, folds.list)
{
  ### Begin Function
  n.sets<-length(preds.list)
  accuracy.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  sensitivity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  specificity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  
  colnames(accuracy.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(sensitivity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(specificity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  
  # use.cores<-6
  # cl.temp<-makeCluster(use.cores)
  # registerDoParallel(cl.temp)
  
  for(k in 1:trials)
  {
    cat("Trial ", k, "\n")
    trial.temp<-foreach(j=1:folds, .combine=append, .packages=c("glmnet"), .export=c("naive.ensembler", "weighted.ensembler", "equal.weighted.ensembler")) %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      ### OUTPUTS FROM FOREACH
      #predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.all<-numeric()
      specificity.all<-numeric()
      sensitivity.all<-numeric()
      
      ### OUTPUTS FROM SOLO ANALYSIS
      predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.solo<-numeric()
      specificity.solo<-numeric()
      sensitivity.solo<-numeric()
      
      ### ACCURACIES AND PREDICTION PROBABILITIES FOR EACH SET OF PREDICTORS
      for(m in 1:n.sets)
      {
        sim.full<-preds.list[[m]]
        
        train.x<-sim.full[,folds.list[[k]]!=j]
        test.x<-sim.full[,folds.list[[k]]==j]
        
        ### RIDGE Regression ###
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
        
        predictions.solo[,m]<-pred.step
        accuracy.solo[m]<-acc.out
        specificity.solo[m]<-spec.out
        sensitivity.solo[m]<-sens.out
      }
      
      #STORE SINGLE CLASSIFIER RESULTS
      accuracy.all[1:n.sets]<-accuracy.solo
      specificity.all[1:n.sets]<-specificity.solo
      sensitivity.all[1:n.sets]<-sensitivity.solo
      
      ### RETURN AND STORE NAIVE ENSEMBLE RESULTS
      naive.ens.out<-naive.ensembler(predictions.solo)
      accuracy.all[n.sets+1]<-naive.ens.out$accuracy.naive
      specificity.all[n.sets+1]<-naive.ens.out$specificity.naive
      sensitivity.all[n.sets+1]<-naive.ens.out$sensitivity.naive
      
      ### RETURN ACCURACY WEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=2)
      accuracy.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+5)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+5)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+5)]<-ens.temp.result$sensitivity.ens
      
      ### RETURN ACCURACY UNWEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=2)
      accuracy.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+9)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+9)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+9)]<-ens.temp.result$sensitivity.ens
      
      output<-list(accuracy.all=accuracy.all, specificity.all=specificity.all, sensitivity.all=sensitivity.all)
      return(list(output))
    }
    
    for(j in 1:folds)
    {
      accuracy.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$accuracy.all
      specificity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$specificity.all
      sensitivity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$sensitivity.all
    }
  }
  # stopCluster(cl.temp)
  return(list(accuracy=accuracy.ensemble.matrix, specificity=specificity.ensemble.matrix, sensitivity=sensitivity.ensemble.matrix))
  
}

ensemble.LDA.sim<-function(preds.list, classes, n.x, folds.list)
{
  ### Begin Function
  n.sets<-length(preds.list)
  accuracy.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  sensitivity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  specificity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  
  colnames(accuracy.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(sensitivity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(specificity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  
  use.cores<-6
  cl.temp<-makeCluster(use.cores)
  registerDoParallel(cl.temp)
  
  for(k in 1:trials)
  {
    cat("Trial ", k, "\n")
    trial.temp<-foreach(j=1:folds, .combine=append, .packages=c("MASS"), .export=c("naive.ensembler", "weighted.ensembler", "equal.weighted.ensembler")) %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      ### OUTPUTS FROM FOREACH
      #predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.all<-numeric()
      specificity.all<-numeric()
      sensitivity.all<-numeric()
      
      ### OUTPUTS FROM SOLO ANALYSIS
      predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.solo<-numeric()
      specificity.solo<-numeric()
      sensitivity.solo<-numeric()
      
      ### ACCURACIES AND PREDICTION PROBABILITIES FOR EACH SET OF PREDICTORS
      for(m in 1:n.sets)
      {
        sim.full<-preds.list[[m]]
        train.x<-sim.full[,folds.list[[k]]!=j]
        test.x<-sim.full[,folds.list[[k]]==j]
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
        
        predictions.solo[,m]<-lda.pred.temp$posterior[,2]
        accuracy.solo[m]<-acc.out
        specificity.solo[m]<-spec.out
        sensitivity.solo[m]<-sens.out
      }
      
      #STORE SINGLE CLASSIFIER RESULTS
      accuracy.all[1:n.sets]<-accuracy.solo
      specificity.all[1:n.sets]<-specificity.solo
      sensitivity.all[1:n.sets]<-sensitivity.solo
      
      ### RETURN AND STORE NAIVE ENSEMBLE RESULTS
      naive.ens.out<-naive.ensembler(predictions.solo)
      accuracy.all[n.sets+1]<-naive.ens.out$accuracy.naive
      specificity.all[n.sets+1]<-naive.ens.out$specificity.naive
      sensitivity.all[n.sets+1]<-naive.ens.out$sensitivity.naive
      
      ### RETURN ACCURACY WEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=2)
      accuracy.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+5)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+5)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+5)]<-ens.temp.result$sensitivity.ens
      
      ### RETURN ACCURACY UNWEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=2)
      accuracy.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+9)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+9)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+9)]<-ens.temp.result$sensitivity.ens
      
      output<-list(accuracy.all=accuracy.all, specificity.all=specificity.all, sensitivity.all=sensitivity.all)
      return(list(output))
    }
    
    for(j in 1:folds)
    {
      accuracy.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$accuracy.all
      specificity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$specificity.all
      sensitivity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$sensitivity.all
    }
  }
  
  return(list(accuracy=accuracy.ensemble.matrix, specificity=specificity.ensemble.matrix, sensitivity=sensitivity.ensemble.matrix))
  
}

ensemble.QDA.sim<-function(preds.list, classes, n.x, folds.list)
{
  ### Begin Function
  n.sets<-length(preds.list)
  accuracy.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  sensitivity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  specificity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  
  colnames(accuracy.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(sensitivity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(specificity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  
  use.cores<-6
  cl.temp<-makeCluster(use.cores)
  registerDoParallel(cl.temp)
  
  for(k in 1:trials)
  {
    cat("Trial ", k, "\n")
    trial.temp<-foreach(j=1:folds, .combine=append, .packages=c("MASS"), .export=c("naive.ensembler", "weighted.ensembler", "equal.weighted.ensembler")) %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      ### OUTPUTS FROM FOREACH
      #predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.all<-numeric()
      specificity.all<-numeric()
      sensitivity.all<-numeric()
      
      ### OUTPUTS FROM SOLO ANALYSIS
      predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.solo<-numeric()
      specificity.solo<-numeric()
      sensitivity.solo<-numeric()
      
      ### ACCURACIES AND PREDICTION PROBABILITIES FOR EACH SET OF PREDICTORS
      for(m in 1:n.sets)
      {
        sim.full<-preds.list[[m]]
        train.x<-sim.full[,folds.list[[k]]!=j]
        test.x<-sim.full[,folds.list[[k]]==j]
        train.df<-as.data.frame(cbind(train.classes, t(train.x)))
        test.df<-as.data.frame(cbind(test.classes, t(test.x)))
        
        ### QDA ### BEWARE NAMES, ONLY CHANGED LDA to QDA
        lda.fit.temp<-qda(train.classes~., data=train.df)
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
        
        predictions.solo[,m]<-lda.pred.temp$posterior[,2]
        accuracy.solo[m]<-acc.out
        specificity.solo[m]<-spec.out
        sensitivity.solo[m]<-sens.out
      }
      
      #STORE SINGLE CLASSIFIER RESULTS
      accuracy.all[1:n.sets]<-accuracy.solo
      specificity.all[1:n.sets]<-specificity.solo
      sensitivity.all[1:n.sets]<-sensitivity.solo
      
      ### RETURN AND STORE NAIVE ENSEMBLE RESULTS
      naive.ens.out<-naive.ensembler(predictions.solo)
      accuracy.all[n.sets+1]<-naive.ens.out$accuracy.naive
      specificity.all[n.sets+1]<-naive.ens.out$specificity.naive
      sensitivity.all[n.sets+1]<-naive.ens.out$sensitivity.naive
      
      ### RETURN ACCURACY WEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=2)
      accuracy.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+5)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+5)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+5)]<-ens.temp.result$sensitivity.ens
      
      ### RETURN ACCURACY UNWEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=2)
      accuracy.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+9)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+9)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+9)]<-ens.temp.result$sensitivity.ens
      
      output<-list(accuracy.all=accuracy.all, specificity.all=specificity.all, sensitivity.all=sensitivity.all)
      return(list(output))
    }
    
    for(j in 1:folds)
    {
      accuracy.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$accuracy.all
      specificity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$specificity.all
      sensitivity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$sensitivity.all
    }
  }
  
  return(list(accuracy=accuracy.ensemble.matrix, specificity=specificity.ensemble.matrix, sensitivity=sensitivity.ensemble.matrix))
  
}


ensemble.KNN.sim<-function(preds.list, classes, opt.k.vect, n.x, folds.list)
{
  # browser()
  ### Begin Function
  n.sets<-length(preds.list)
  accuracy.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  sensitivity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  specificity.ensemble.matrix<-matrix(ncol=(2^n.sets+4), nrow=folds*trials)
  
  colnames(accuracy.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(sensitivity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  colnames(specificity.ensemble.matrix)<-c("D0", "D1", "D2", "Naive", "W:D0-D1", "W:D0-D2", "W:D1-D2", "W:D0-D1-D2", "D0-D1", "D0-D2", "D1-D2", "D0-D1-D2")
  
  use.cores<-6
  cl.temp<-makeCluster(use.cores)
  registerDoParallel(cl.temp)
  
  for(k in 1:trials)
  {
    cat("Trial ", k, "\n")
    trial.temp<-foreach(j=1:folds, .combine=append, .packages=c("class"), .export=c("naive.ensembler", "weighted.ensembler", "equal.weighted.ensembler")) %dopar%
    {
      ### LENGTHS
      n.train<-length(which(folds.list[[k]]!=j))
      n.test<-length(which(folds.list[[k]]==j))
      
      ### CLASSES AND COUNTS
      train.classes<-classes[folds.list[[k]]!=j]
      test.classes<-classes[folds.list[[k]]==j]
      N0<-length(which(test.classes==0))
      N1<-length(which(test.classes==1))
      
      ### OUTPUTS FROM FOREACH
      #predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.all<-numeric()
      specificity.all<-numeric()
      sensitivity.all<-numeric()
      
      ### OUTPUTS FROM SOLO ANALYSIS
      predictions.solo<-matrix(ncol=n.sets, nrow=n.test)
      accuracy.solo<-numeric()
      specificity.solo<-numeric()
      sensitivity.solo<-numeric()
      
      ### ACCURACIES AND PREDICTION PROBABILITIES FOR EACH SET OF PREDICTORS
      for(m in 1:n.sets)
      {
        sim.full<-preds.list[[m]]
        
        train.x<-sim.full[,folds.list[[k]]!=j]
        test.x<-sim.full[,folds.list[[k]]==j]
        train.df<-as.data.frame(cbind(train.classes, t(train.x)))
        test.df<-as.data.frame(cbind(test.classes, t(test.x)))
        
        knn.test<-knn(t(train.x), t(test.x), train.classes, k=opt.k.vect[m], prob = TRUE)
        
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
        
        predictions.solo[,m]<-probs.temp
        accuracy.solo[m]<-acc.out
        specificity.solo[m]<-spec.out
        sensitivity.solo[m]<-sens.out
      }
      
      #STORE SINGLE CLASSIFIER RESULTS
      accuracy.all[1:n.sets]<-accuracy.solo
      specificity.all[1:n.sets]<-specificity.solo
      sensitivity.all[1:n.sets]<-sensitivity.solo
      
      ### RETURN AND STORE NAIVE ENSEMBLE RESULTS
      naive.ens.out<-naive.ensembler(predictions.solo)
      accuracy.all[n.sets+1]<-naive.ens.out$accuracy.naive
      specificity.all[n.sets+1]<-naive.ens.out$specificity.naive
      sensitivity.all[n.sets+1]<-naive.ens.out$sensitivity.naive
      
      ### RETURN ACCURACY WEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=2)
      accuracy.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+2):(n.sets+4)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-weighted.ensembler(accuracy.solo, predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+5)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+5)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+5)]<-ens.temp.result$sensitivity.ens
      
      ### RETURN ACCURACY UNWEIGHTED ENSEMBLE FOR ALL COMBINATIONS
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=2)
      accuracy.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+6):(n.sets+8)]<-ens.temp.result$sensitivity.ens
      
      ens.temp.result<-equal.weighted.ensembler(predictions.solo, ens.size=n.sets)
      accuracy.all[(n.sets+9)]<-ens.temp.result$accuracy.ens
      specificity.all[(n.sets+9)]<-ens.temp.result$specificity.ens
      sensitivity.all[(n.sets+9)]<-ens.temp.result$sensitivity.ens
      
      output<-list(accuracy.all=accuracy.all, specificity.all=specificity.all, sensitivity.all=sensitivity.all)
      return(list(output))
    }
    
    for(j in 1:folds)
    {
      accuracy.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$accuracy.all
      specificity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$specificity.all
      sensitivity.ensemble.matrix[folds*(k-1)+j,]<-trial.temp[[j]]$sensitivity.all
    }
  }
  
  return(list(accuracy=accuracy.ensemble.matrix, specificity=specificity.ensemble.matrix, sensitivity=sensitivity.ensemble.matrix))
}









