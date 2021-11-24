
####### Anexo del ejercicio 1#########################
############### Random Forest:###############


# Calculamos nuestros errores obtenidos anteriormente:  
er_val1=function(materr){
  e3=matrix(NA,ncol=3, nrow=1)
  colnames(e3)=c("NO","Yes","Global")
  rownames(e3)=c("Train")
  aux=colMeans(materr)
  e3[1,1]=aux[1]
  e3[1,2]=aux[2]
  e3[1,3]=aux[3]
  return(e3)
}

er_val2=function(materr){
  e4=matrix(NA,ncol=3, nrow=1)
  colnames(e4)=c("NO","Yes","Global")
  rownames(e4)=c("Test")
  aux=colMeans(materr)
  e4[1,1]=aux[4]
  e4[1,2]=aux[5]
  e4[1,3]=aux[6]
  return(e4)
}

# Agregamos los paquetes:
library(MASS)
library(randomForest)
library(gbm)
library(caret)

# Cargamos el dataset:
bd=rbind(Pima.tr,Pima.te)

# Fijamos la semilla:
set.seed(1859)

# Realizamos nuestro primer modelo con todas las variables:
model <- randomForest(type ~ ., data=bd, ntree=3000, proximity=TRUE)
model

# Comprobamos la cantidad de árboles que necesitamos:
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "No", "Yes"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
          model$err.rate[,"No"], 
          model$err.rate[,"Yes"]))

# Graficamos:
ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

# Comprobamos el mejor valor del parámetro mtry:
oob.values <- vector(length=7)
for(i in 1:7) {
  temp.model <- randomForest(type ~ ., data=bd, mtry=i, ntree=1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values

# Buscamos el error mínimo:
min(oob.values)

# Buscamos el valor óptimo de mtry:
which(oob.values == min(oob.values))

# Ajustamos el model con los mejores parámetros:
set.seed(1859)
model <- randomForest(type ~ ., 
                      data=bd,
                      ntree=1000, 
                      proximity=TRUE, 
                      mtry=which(oob.values == min(oob.values)),importance=T,strata=bd$type)
model

# Comprobamos la importancia de las variables:
varImpPlot(model)

# Definimos la matriz de los errores y hacemos un ciclo que calcula los errores:
Nrep=100
set.seed(1859)
materr1=matrix(NA,ncol=6, nrow=Nrep)
colnames(materr1)=c("trg_No","trg_Yes","trg_Global", "tst_NO","tst_YES","tst_Global")
for(irep in 1:Nrep){
  train=createDataPartition(bd$type, p=0.7)$Resample1
  model_Train=randomForest(type~., data=bd[train,],ntree=1000, proximity=TRUE, 
                           mtry=which(oob.values == min(oob.values)),importance=T)
  materr1[irep,1]=model_Train$confusion[1,3]
  materr1[irep,2]=model_Train$confusion[2,3]
  materr1[irep,3]=(model_Train$confusion[1,2]+model_Train$confusion[2,1])/
    sum(model_Train$confusion[,1:2])
  t3=table(bd[-train,]$type,predict(model_Train,newdata = bd[-train,]))
  materr1[irep,4]=t3[1,2]/(t3[1,1]+t3[1,2]) 
  materr1[irep,5]=t3[2,1]/(t3[2,1]+t3[2,2])
  materr1[irep,6]=(t3[1,2]+t3[2,1])/sum(t3)
}

# Guardamos nuestros errores en las variables e3 y e4:
e3=er_val1(materr1)
e4=er_val2(materr1)
e3
e4

## Realizamos el segundo modelo de acuerdo a las importancias:
set.seed(1859)

# Ajustamos el modelo:
model=randomForest(type~glu+bmi+age+ped,data = bd, ntre=3000, proximity=T, importance=T)
model

# Comprobamos el mejor valor del parámetro mtry:
oob.values <- vector(length=4)
for(i in 1:4) {
  temp.model <- randomForest(type ~ glu+bmi+age+ped, data=bd, mtry=i, ntree=1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values

# Buscamos el error mínimo:
min(oob.values)

# Buscamos el valor óptimo de mtry:
which(oob.values == min(oob.values))

# Fijamos la semilla:
set.seed(1859)

# Ajustamos el modelo con los parámetros encontrados:
model=randomForest(type~glu+bmi+age+ped,data = bd, ntre=1000,mtry= which(oob.values == min(oob.values)),proximity=T, importance=T)
model

# Definimos la matriz de los errores y hacemos un ciclo que calcula los errores:
materr1=matrix(NA,ncol=6, nrow=Nrep)
colnames(materr1)=c("trg_No","trg_Yes","trg_Global", "tst_NO","tst_YES","tst_Global")
set.seed(1859)
for(irep in 1:Nrep){
  train=createDataPartition(bd$type, p=0.7)$Resample1
  model_Train=randomForest(type ~ glu+bmi+age+ped, data=bd[train,],ntree=1000, proximity=TRUE, 
                           mtry=2,importance=T)
  materr1[irep,1]=model_Train$confusion[1,3]
  materr1[irep,2]=model_Train$confusion[2,3]
  materr1[irep,3]=(model_Train$confusion[1,2]+model_Train$confusion[2,1])/
    sum(model_Train$confusion[,1:2])
  t3=table(predict(model_Train,newdata = bd[-train,]),bd[-train,]$type)
  materr1[irep,4]=t3[1,2]/(t3[1,1]+t3[1,2]) 
  materr1[irep,5]=t3[2,1]/(t3[2,1]+t3[2,2])
  materr1[irep,6]=(t3[1,2]+t3[2,1])/sum(t3)
}

# Guardamos los errores en las variables e3 y e4:
e3=er_val1(materr1)
e4=er_val2(materr1)
e3
e4

# Realizamos un último ajuste, esta vez agregando la varaible npreg:

# Fijamos la semilla:
set.seed(1859)

# Comprobamos el mejor valor del parámetro mtry:
oob.values <- vector(length=5)
for(i in 1:5) {
  temp.model <- randomForest(type ~ glu+bmi+age+ped+npreg, data=bd, mtry=i, ntree=1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values

# Buscamos el menor error:
min(oob.values)

# Buscamos los valores óptimos:
which(oob.values == min(oob.values))

# Fijamos semilla:
set.seed(1859)

# Tomamos el modelo con los parámetros encontrados:
model=randomForest(type~glu+bmi+age+ped+npreg,data = bd, ntree=1000,mtry=2, proximity=T, importance=T)
model

# Definimos la matriz de los errores y hacemos un ciclo que calcula los errores:
materr1=matrix(NA,ncol=6, nrow=200)
colnames(materr1)=c("trg_No","trg_Yes","trg_Global", "tst_NO","tst_YES","tst_Global")
set.seed(1859)
for(irep in 1:200){
  train=createDataPartition(bd$type, p=0.7)$Resample1
  model_Train=randomForest(type~glu+bmi+age+ped+npreg, data=bd[train,],ntree=1000, proximity=TRUE, 
                           mtry=2,importance=T)
  materr1[irep,1]=model_Train$confusion[1,3]
  materr1[irep,2]=model_Train$confusion[2,3]
  materr1[irep,3]=(model_Train$confusion[1,2]+model_Train$confusion[2,1])/
    sum(model_Train$confusion[,1:2])
  t3=table(bd[-train,]$type,predict(model_Train,newdata = bd[-train,]))
  materr1[irep,4]=t3[1,2]/(t3[1,1]+t3[1,2]) 
  materr1[irep,5]=t3[2,1]/(t3[2,1]+t3[2,2])
  materr1[irep,6]=(t3[1,2]+t3[2,1])/sum(t3)
}

# Guardamos nuestros errores en las variables e3 y e4:
e3=er_val1(materr1)
e4=er_val2(materr1)
e3
e4

#######Boosting###############

# Copiamos la base de datos:
df=bd

# Cambiamos la variable de respuesta por 0 y 1: 
df$type =ifelse(bd$type=="No",0,1)

# Fijamos semilla:
set.seed(1859)

# Creamos las varaibles donde guardaremos los errores:
app.err.500=double(7)
cv10.err.500=double(7)
app.err.1000=double(7)
cv10.err.1000=double(7)

# Realizamos dos ciclos para determinar el mejor numero de iteraciones:
for(intdepth in 1:7){
  set.seed(1+intdepth)
  Bs.bs.500=(gbm(type~.,data=df,distribution="bernoulli",
                 n.trees=500,interaction.depth=intdepth, cv.folds = 10))
  app.err.500[intdepth]=Bs.bs.500$train.error[500]
  cv10.err.500[intdepth]=Bs.bs.500$cv.error[500]
}

for(intdepth in 1:7){
  set.seed(1+intdepth)
  Bs.bs.1000=(gbm(type~.,data=df,distribution="bernoulli",
                  n.trees=1000,interaction.depth=intdepth, cv.folds = 10))
  app.err.1000[intdepth]=Bs.bs.1000$train.error[1000]
  cv10.err.1000[intdepth]=Bs.bs.1000$cv.error[1000]
}

# Graficamos los errores calculados:
par(mfrow=c(1,1))
matplot(1:intdepth, cbind(app.err.500, cv10.err.500,
                          app.err.1000, cv10.err.1000),pch=19,cex=1.0, 
        col=c("black", "red", "green","blue"), lty=c(1,1,2,2),
        type="b",ylab="Mean square error")
legend("bottomleft", legend=c("App 500", "cv10-500",
                              "App 1000", "cv10-1000"), pch=19, cex=.6,
       col=c("black", "red", "green","blue"))

# Determinamos el número de árboles óptimo con idepth=4 y shrink=0.01:
set.seed(1859)
model_Train=(gbm(type~.,data=df,distribution="bernoulli",
                 n.trees=500, interaction.depth=4,shrinkage=0.01,cv.folds = 10))
ntree_opt_cv=gbm.perf(model_Train,method = "cv")
ntree_opt_cv

# Ajustamos nuevamente pero con el número de árboles que nos indican como mejor:
set.seed(1859)
model_Train=(gbm(type~.,data=df,distribution="bernoulli",
                 n.trees=302, interaction.depth=4,shrinkage=0.01,cv.folds = 10))
summary(model_Train)

gbm.biop.test = predict(model_Train, newdata=df, type="response", n.trees=302)
gbm.class =ifelse(gbm.biop.test<0.4,"No", "Yes")
(t3=table(bd$type,gbm.class))


# Definimos la matriz de los errores y hacemos un ciclo que calcula los errores:
materr=matrix(NA,ncol=6, nrow=1000)
colnames(materr)=c("trg_No","trg_Yes","trg_Global", "tst_NO","tst_YES","tst_Global")
set.seed(1859)
for(irep in 1:1000){
  train=createDataPartition(df$type, p=0.7)$Resample1
  model_Train=(gbm(type~.,data=df[train,],distribution="bernoulli",
                   n.trees=302, interaction.depth=4,shrinkage=0.01))
  gbm.biop.test = predict(model_Train, newdata=df[train,], type="response", n.trees=302)
  gbm.class =ifelse(gbm.biop.test<0.5,"No", "Yes")
  (t3=table(df[train,]$type,gbm.class))
  materr[irep,1]=t3[1,2]/(t3[1,1]+t3[1,2]) 
  materr[irep,2]=t3[2,1]/(t3[2,1]+t3[2,2])
  materr[irep,3]=(t3[1,2]+t3[2,1])/sum(t3)
  gbm.biop.test = predict(model_Train, newdata=df[-train,], type="response", n.trees=302)
  gbm.class =ifelse(gbm.biop.test<0.5,"No", "Yes")
  (t3=table(df[-train,]$type,gbm.class))
  materr[irep,4]=t3[1,2]/(t3[1,1]+t3[1,2]) 
  materr[irep,5]=t3[2,1]/(t3[2,1]+t3[2,2])
  materr[irep,6]=(t3[1,2]+t3[2,1])/sum(t3)
}

# Guardamos los errores:
e3=er_val1(materr)
e4=er_val2(materr)
e3
e4

#Selecciono n.trees=500, intdepth=4.
set.seed(1859)
model_Train=(gbm(type~glu+age+ped+bmi,data=df,distribution="bernoulli",
                 n.trees=500, interaction.depth=4,shrinkage=0.01,cv.folds = 10))

ntree_opt_cv=gbm.perf(model_Train,method = "cv")
ntree_opt_cv

set.seed(1859)
model_Train=(gbm(type~glu+age+ped+bmi,data=df,distribution="bernoulli",
                 n.trees=307, interaction.depth=4,shrinkage=0.01))

gbm.biop.test = predict(model_Train, newdata=df, type="response", n.trees=307)
gbm.class =ifelse(gbm.biop.test<0.5,"No", "Yes")
(t3=table(df$type,gbm.class))


# Definimos la matriz de los errores y hacemos un ciclo que calcula los errores:
materr=matrix(NA,ncol=6, nrow=1000)
colnames(materr)=c("trg_No","trg_Yes","trg_Global", "tst_NO","tst_YES","tst_Global")

set.seed(1859)
for(irep in 1:1000){
  train=createDataPartition(df$type, p=0.7)$Resample1
  model_Train=(gbm(type~glu+age+ped+bmi,data=df[train,],distribution="bernoulli",
                   n.trees=307, interaction.depth=4,shrinkage=0.01))
  gbm.biop.test = predict(model_Train, newdata=df[train,], type="response", n.trees=307)
  gbm.class =ifelse(gbm.biop.test<0.5,"No", "Yes")
  (t3=table(df[train,]$type,gbm.class))
  materr[irep,1]=t3[1,2]/(t3[1,1]+t3[1,2]) 
  materr[irep,2]=t3[2,1]/(t3[2,1]+t3[2,2])
  materr[irep,3]=(t3[1,2]+t3[2,1])/sum(t3)
  gbm.biop.test = predict(model_Train, newdata=df[-train,], type="response", n.trees=307)
  gbm.class =ifelse(gbm.biop.test<0.5,"No", "Yes")
  (t3=table(df[-train,]$type,gbm.class))
  materr[irep,4]=t3[1,2]/(t3[1,1]+t3[1,2]) 
  materr[irep,5]=t3[2,1]/(t3[2,1]+t3[2,2])
  materr[irep,6]=(t3[1,2]+t3[2,1])/sum(t3)
}

# Guardamos los errores:
e3=er_val1(materr)
e4=er_val2(materr)
e3
e4

# Graficamos:
par(mfrow=c(1,1))
matplot(1:3,cbind(c(0.11008,0.4288,0.2163),c(0.15038,0.3713,0.2240),c(0.10764,
                                                                      0.4218,0.2123),c(0.11368,0.4226,0.2167),c(0.1146,0.4252,0.2194),c(0.1166,0.3965,0.2107)),pch=20, cex=1.7,xaxt='n', ann=T,
        col=c("black","blue","red","green","orange","purple"),
        type="b",ylab="Error", xlab=" Clases",lty = 2, lwd = ,main=" Test Errors")
+axis(side=1,at=c(1,2,3),labels=c("No","Yes","Global"))

legend("topleft", legend=c("LDA_Test","NB_Test","LR_Test","SVM_Test","RF_Test","Boost_Test") ,
       pch=19, cex=.8, col=c("black","blue","red","green","orange","purple"))

###### Ejercicio 3:
names(crashdata)
data=crashdata[,c(1,4,5,6,7,9,11,13)]
names(data)


app.err.500=double(7)
cv10.err.500=double(7)
app.err.1000=double(7)
cv10.err.1000=double(7)

t=proc.time()
for(intdepth in 1:7){
  set.seed(1+intdepth)
  Bs.bs.500=(gbm(log(Crash_Score)~.,data=data,distribution="gaussian",
                 n.trees=500,interaction.depth=intdepth, cv.folds = 10))
  app.err.500[intdepth]=Bs.bs.500$train.error[500]
  cv10.err.500[intdepth]=Bs.bs.500$cv.error[500]
}

proc.time()-t
# user  system elapsed 
# 3.972   0.261  34.652 

t=proc.time()
for(intdepth in 1:7){
  set.seed(1+intdepth)
  Bs.bs.1000=(gbm(log(Crash_Score)~.,data=data,distribution="gaussian",
                 n.trees=1000,interaction.depth=intdepth, cv.folds = 10))
  app.err.1000[intdepth]=Bs.bs.1000$train.error[1000]
  cv10.err.1000[intdepth]=Bs.bs.1000$cv.error[1000]
}
proc.time()-t
# user  system elapsed 
# 7.705   0.386  48.146 
Bs.bs.1000$train.error[1000]
cbind(cv10.err.500, cv10.err.1000)

par(mfrow=c(1,1))
matplot(1:intdepth, cbind(app.err.500, cv10.err.500,
                          app.err.1000, cv10.err.1000),pch=19,cex=1.0, 
        col=c("black", "red", "black","red"), lty=c(1,1,2,2),
        type="b",ylab="Mean square error")
legend("topright", legend=c("App 500", "cv10-500",
                            "App 1000", "cv10-1000"), pch=19, cex=.6,
       col=c("black", "red", "black","red"))

which.min(cv10.err.500)
sort(cv10.err.500)

#Selecciono n.trees=500, intdepth=5.
Bs.bs.500Best=(gbm(log(Crash_Score)~.,data=data,distribution="gaussian",
                   n.trees=500,interaction.depth=2))
summary(Bs.bs.500Best)
Bs.bs.500Best
errors
(errors[5,1]=Bs.bs.500Best$train.error[500])
#=
(mean((Bs.bs.500Best$fit-Boston$medv)^2))

(errors[5,3]=Bs.bs.500Best$cv.error[500])

#Boosting interaction.depth=1 default
Bs.bs.500dft=(gbm(medv~.,data=Boston,distribution="gaussian",
                  n.trees=500, cv.folds = 10))
summary(Bs.bs.500dft)
Bs.bs.500dft$interaction.depth
(errors[4,1]=Bs.bs.500dft$train.error[500])
(errors[4,3]=Bs.bs.500dft$cv.error[500])
print(errors, digits=3)

# Cargamos la librerías
library(randomForest)
library(readr)
####### Anexo ejericio 2:#############

# Cargamos la base de datos:
Glucose <- read_csv("C:/Users/Family/Desktop/Mat Niye/estadistica/aprendizaje estadistico/tarea 4/ejercicio2/Glucose1.txt")

Glucosa<- Glucose[,-1]

Glucosa$Class<- factor(Glucosa$Class)
summary(Glucosa)
clas_real=table(Glucosa$Class)

#Random forest :

# Selección del mtry:

par(mfrow=c(1,1))
oob.err=double(5)
app.err=double(5)
set.seed(1234)
t=proc.time()
for(m in 1:5){
  set.seed(2*m)
  RFp_glucosa=randomForest(x=Glucosa[,-6],y=Glucosa$Class, data=Glucosa,
                           ntree = 500, mtry=m,strata=Glucosa$Class) 
  oob.err[m]=1-sum(diag(table(RFp_glucosa$predicted,Glucosa$Class)))/sum(table(RFp_glucosa$predicted,Glucosa$Class)) #predicted oob error ~Test error
  app.err[m]=1-sum(diag(table(predict(RFp_glucosa,Glucosa),Glucosa$Class)))/sum(table(predict(RFp_glucosa,Glucosa),Glucosa$Class)) #apparent
  cat(m,"")
}
proc.time()-t

oob.err
app.err

cbind(app.err,oob.err) 
which.min(oob.err)
oob.err[which.min(oob.err)]

matplot(1:5, 
        cbind(app.err, oob.err),col=c("black","red"),
        pch=19,type="b",ylab="Mean square error")
legend("topright", legend=c("App506", "oob"), pch=19,
       cex=.8,col=c("black","red"))# Se puede hacer la seleccion de 1,2,3 en nuestro caso se escoje 3.

# Modelo randomforest:

# Predicciones sobre el conjunto de validación:
B=2000
aux2_4=matrix(NA,ncol = 4, nrow = 2000)
set.seed(126)
for (i in 1:1000) {
  train=createDataPartition(Glucosa$Class, p=.7, list=FALSE)
  RF_glucosa=randomForest(x=Glucosa[train,][,-6],y=Glucosa[train,]$Class, data=Glucosa[train,],
                          ntree = 500, mtry=2,strata=Glucosa[train,]$Class)
  taux=table(Glucosa$Class[-train],predict(RF_glucosa, Glucosa[-train,], type = "class"))
  for(j in 1:3) aux2_4[i,j]=1- taux[j,j]/sum(taux[j,])
  aux2_4[i,4]=1- (taux[1,1]+taux[2,2]+taux[3,3])/sum(taux)
}

# Errores de prediccion:
mean(aux2_4[,1]) # Error clase 1
mean(aux2_4[,2]) # Error clase 2
mean(aux2_4[,3]) # Error clase 3
mean(aux2_4[,4]) # Error clase 4

# Error aparente:
table(predict(RF_glucosa,Glucosa),Glucosa$Class) #Se hace con una repetición es 0

#######Boosting#####

library(h2o)
h2o.init()

# Importamos la base de datos en el formato de h2o:
Glucoseh2o <- h2o.importFile("C:/Users/Family/Desktop/Mat Niye/estadistica/aprendizaje estadistico/tarea 4/ejercicio2/Glucose1.txt")

Glucosah2o<- Glucoseh2o[,-1]

Glucosah2o$Class<- as.factor(Glucosah2o$Class)
summary(Glucosah2o)
clas_real=table(Glucosah2o$Class)

summary(Glucosa)

# Selección de árboles:

# Fijamos los predictores y variables explicativas; fijamos los factores:
predictors <- c("Weight", "Fglucose", "GlucoseInt", "InsulinResp", "InsulineResist")
response <- "Class"

# Construimos el modelo de entrenamiento:
gluc_gbm <- h2o.gbm(x = predictors,
                    y = response,
                    nfolds = 4,
                    seed = 1231,
                    keep_cross_validation_predictions = TRUE,
                    training_frame = Glucosah2o)

# Evaluamos el desempeño:
perf <- h2o.performance(gluc_gbm)
perf #Error aparente 0

# REpeared training test:
B=2000
aux2_4=matrix(NA,ncol = 4, nrow = 2000)
set.seed(1234)
for (i in 1:2000) {
  train=createDataPartition(Glucosa$Class, p=.7, list=FALSE)
  trainh2o=Glucosah2o[train,]
  gluc_gbm <- h2o.gbm(x = predictors,
                      y = response,
                      nfolds = 4,
                      seed = 121,
                      keep_cross_validation_predictions = TRUE,
                      training_frame = trainh2o)
  taux=h2o.table(Glucosah2o$Class[-train],predict(gluc_gbm, Glucosah2o[-train,], type = "class")[,1],dense = FALSE)
  taux=taux[,2:4]
  for(j in 1:3) aux2_4[i,j]=1- taux[j,j]/sum(taux[j,])
  aux2_4[i,4]=1- (taux[1,1]+taux[2,2]+taux[3,3])/sum(taux)
}

aux2_4
mean(aux2_4[,1])# Error clase1
mean(aux2_4[,2])# Error clase2
mean(aux2_4[,3])# Error clase3
mean(aux2_4[,4])# Error global


######### Anexo Ejercicio 3:##############

# Random forest:
View(crashdata)
crashdata2<-crashdata[,-c(2,3,8,10,12,14)]

par(mfrow=c(1,1))
oob.err=double(7)
app.err=double(7)

t=proc.time()
for(m in 1:7){
  set.seed(2*m)
  RF_Crash=randomForest(Crash_Score~ Time_of_Day + Rd_Feature + Rd_Class + Rd_Character+ Rd_Surface + Light + Traffic_Control, data=crashdata,
                        ntree = 50, mtry=m)  
  oob.err[m]=mean((RF_Crash$predicted- crashdata$Crash_Score)^2)# predicted oob error ~Test error
  app.err[m]=mean((predict(RF_Crash, crashdata2)-crashdata2$Crash_Score)^2) # apparent
  cat(m,"")
}
proc.time()-t

oob.err
app.err
RF_Crash$mse

mean((exp(RF_Crash$predicted)- crashdata$Crash_Score)^2)
crashdata$Crash_Score
cbind(app.err,oob.err) 
which.min(oob.err)
oob.err[which.min(oob.err)]

matplot(1:7, 
        cbind(app.err, oob.err),col=c("black","red"),
        pch=19,type="b",ylab="Mean square error")
legend("topright", legend=c("App506", "oob"), pch=19,
       cex=.8,col=c("black","red"))

RF_Crash=randomForest(Crash_Score~ Time_of_Day + Rd_Feature + Rd_Class + Rd_Character+ Rd_Surface + Light + Traffic_Control, data=crashdata,
                      ntree = 500, mtry=7)# En 500 no cambia mucho

mean((exp(RF_Crash$predicted)-crashdata$Crash_Score)^2)

RF_Crash$mse


RF2_Crash=randomForest(log(Crash_Score)~ Time_of_Day + Rd_Feature + Rd_Class + Rd_Character+ Rd_Surface + Light + Traffic_Control, data=crashdata,
                       ntree = 50, mtry=3)

# Hacer repeated training test:
set.seed(1234)
error_test=rep(0,6)
error_apa=rep(0,6)
testerrout=rep(0,6)
for (B in c(10,20,100,500,1000,2000)) {
  oob_err=rep(0,B)
  app_err=rep(0,B)
  testerr=rep(0,B)
  for (k in 1:B) {
    dt = sort(sample(nrow(crashdata), nrow(crashdata)*.7))
    traincrash<-crashdata[dt,]
    testcrash <-crashdata[-dt,]
    
    RF_Crash=randomForest(Crash_Score~ Time_of_Day + Rd_Feature + Rd_Class + Rd_Character+ Rd_Surface + Light + Traffic_Control, data=traincrash,
                          ntree = 50, mtry=3)
    oob_err[k]=mean((RF_Crash$predicted- traincrash$Crash_Score)^2)#predicted oob error ~Test error
    app_err[k]=mean((predict(RF_Crash, traincrash)-traincrash$Crash_Score)^2) #training
    testerr[k]=mean((predict(RF_Crash, testcrash)-testcrash$Crash_Score)^2) #test
    cat(k,"")
    
  }
  if(B==10){
    error_apa[1]=mean(app_err)
    error_test[1]=mean(oob_err)
    testerrout[1]=mean(testerr)
  }
  if(B==20){
    error_apa[2]=mean(app_err)
    error_test[2]=mean(oob_err)
    testerrout[2]=mean(testerr)
  }
  if(B==100){
    error_apa[3]=mean(app_err)
    error_test[3]=mean(oob_err)
    testerrout[3]=mean(testerr)
  }
  if(B==500){
    error_apa[4]=mean(app_err)
    error_test[4]=mean(oob_err)
    testerrout[4]=mean(testerr)
  }
  if(B==1000){
    error_apa[5]=mean(app_err)
    error_test[5]=mean(oob_err)
    testerrout[5]=mean(testerr)
    
  }
  if(B==2000){
    error_apa[6]=mean(app_err)
    error_test[6]=mean(oob_err)
    testerrout[6]=mean(testerr)
    
  }
}  

error_test # Errores de Oob
error_apa # Errores de Training
testerrout # Errores test(T-T)
# Se toma con mil las repeticiones.

############Boosting##############

crashdata2=crashdata[,-c(2,3,8,10,12,14)]
app.err.500=double(7)
cv10.err.500=double(7)
app.err.1000=double(7)
cv10.err.1000=double(7)
#Numero para interaction.depth
t=proc.time()
for(intdepth in 1:7){
  set.seed(2+intdepth)
  Bs.bs.500=(gbm(Crash_Score ~.,distribution = "gaussian",data = crashdata2, shrinkage = 0.1,n.trees = 50,
                 interaction.depth=intdepth,cv.folds = 5))
  app.err.500[intdepth]=Bs.bs.500$train.error[50]
  cv10.err.500[intdepth]=Bs.bs.500$cv.error[50]
}
app.err.500
cv10.err.500
proc.time()-t
# Training-test con B=1000
set.seed(1243)
err_trainb=rep(0,500)
err_testb=rep(0,500)
for (i in 1:500) {
  dt = sort(sample(nrow(crashdata2), nrow(crashdata2)*.7))
  traincrash<-crashdata2[dt,]
  testcrash <-crashdata2[-dt,]
  gbm1=gbm(Crash_Score ~.,distribution = "gaussian",data = traincrash, shrinkage = 0.1,n.trees = 100,interaction.dept=4)
  err_testb[i]=mean((predict(gbm1,testcrash)-testcrash$Crash_Score)**2)
  err_trainb[i]=mean((predict(gbm1,traincrash)-traincrash$Crash_Score)**2)
  
}
e1=mean(err_trainb)# Error TRAIN
e2=mean(err_testb)# Error TEST
gbm1=gbm(Crash_Score ~.,distribution = "gaussian",data = crashdata2, shrinkage = 0.1,n.trees = 500,interaction.dept=4)
mean((predict(gbm1,crashdata2)-crashdata2$Crash_Score)**2)# Error aparente

# Boosting Gamma
library(mboost)


model  <- mboost( Crash_Score ~ ., data = crashdata2,
                  baselearner = "btree", family = GammaReg(), 
                  control = boost_control(mstop = 1000))
mean((exp(predict(model,crashdata2))-crashdata2$Crash_Score)**2)
# Repeated training-test
set.seed(1243)
err_trainb=rep(0,1000)
err_testb=rep(0,1000)
for (i in 1:1000) {
  dt = sort(sample(nrow(crashdata2), nrow(crashdata2)*.7))
  traincrash<-crashdata2[dt,]
  testcrash <-crashdata2[-dt,]
  model  <- mboost( Crash_Score ~ ., data = traincrash,
                    baselearner = "btree", family = GammaReg(), 
                    control = boost_control(mstop = 500))
  err_testb[i]=mean((exp(predict(model,testcrash))-testcrash$Crash_Score)**2)
  err_trainb[i]=mean((exp(predict(model,traincrash))-traincrash$Crash_Score)**2)
  
}
e1g=mean(err_trainb)
e2g=mean(err_testb)
err_testb


