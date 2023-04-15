library(knitr)
library(tidyverse)
library(ggplot2)
library(mice)
library(lattice)
library(reshape2)
library(DataExplorer)
library(corrplot)
library(caret)


#Import data
data=default_of_credit_card_clients_1_

#Correlation plot
corrplot(cor(data),method = "shade",tl.col = "black",tl.pos	="lt",tl.cex = 0.5)

#Conversion categorical data in factors
data$`SEX(x2)`=as.factor(data$`SEX(x2)`)
data$`EDUCATION(x3)`=as.factor(data$`EDUCATION(x3)`)
data$`MARRIAGE(x4)`=as.factor(data$`MARRIAGE(x4)`)
data$`PAY_0(x6)`=as.factor(data$`PAY_0(x6)`)
data$`PAY_2(x7)`=as.factor(data$`PAY_2(x7)`)
data$`PAY_3(x8)`=as.factor(data$`PAY_3(x8)`)
data$`PAY_4(x9)`=as.factor(data$`PAY_4(x9)`)
data$`PAY_5(x10)`=as.factor(data$`PAY_5(x10)`)
data$`PAY_6(x11)`=as.factor(data$`PAY_6(x11)`)
data$`default payment next month(y)`=as.factor(data$`default payment next month(y)`)
str(data)

#Exploratory Data Analysis
dim(data)
str(data)
summary(data)
introduce(data)
count(data, vars = data$`EDUCATION(x3)`)
count(data, vars = data$`MARRIAGE(x4)`)

#replace 0's with NAN, replace others too
data$`EDUCATION(x3)`[data$`EDUCATION(x3)` == 0] <- 4
data$`EDUCATION(x3)`[data$`EDUCATION(x3)`== 5] <- 4
data$`EDUCATION(x3)`[data$`EDUCATION(x3)`== 6] <- 4
data$`MARRIAGE(x4)` [data$`MARRIAGE(x4)` == 0] <- 3

count(data, vars = data$`MARRIAGE(x4)`)
count(data, vars = data$`EDUCATION(x3)`)
plot_histogram(data)

#Feature Engineering
#deleting columns

data_new = select(data, -one_of('ID','AGE(x5)', 'BILL_AMT2(x13)','BILL_AMT3(x14)','BILL_AMT4(x15)','BILL_AMT5(x16)','BILL_AMT6(x17)'))

head(data_new)

#Spliting data into two halves TRAIING and TEST data
#sample of 70% rows
data2 = sort(sample(nrow(data_new), nrow(data_new)*.7))

#creating training data set by selecting the output row values
train = data_new[data2,]

#creating test data set by not selecting the output row values
test = data_new[-data2,-18]

dim(train)
dim(test)
View(test)

##//Model Development//##

# 1.Fitting of LOGISTIC REGRESSION MODEL
log.model = glm(train$`default payment next month(y)` ~., data = train, family = "binomial")
summary(log.model)

log.model$xlevels[["PAY_2(x7)"]] <- union(log.model$xlevels[["PAY_2(x7)"]], levels(test$`PAY_2(x7)`))
log.model$xlevels[["PAY_4(x9)"]] <- union(log.model$xlevels[["PAY_4(x9)"]], levels(test$`PAY_4(x9)`))
log.model$xlevels[["PAY_5(x10)"]] <- union(log.model$xlevels[["PAY_5(x10)"]], levels(test$`PAY_5(x10)`))

# Prediction using logistic regression model, probablilities obtained
log.predictions = predict(log.model, test, type="response")

yhat=ifelse(log.predictions<0.5,0,1)
y=data_new[-data2,18]
y=unlist(y)
length(yhat)
length(y)

## Look at probability output
head(log.predictions)
head(yhat)
head(y)


#Confusion Matrix
yhat1=as.factor(yhat)
y1=as.factor(y)
CLR <- confusionMatrix(data = yhat1, reference = y1);CLR
CLRt=CLR$table

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Class1', cex=2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Class2', cex=2)
  text(125, 370, 'Predicted', cex=2.2, srt=90, font=2)
  text(245, 450, 'Actual', cex=2.2, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Class1', cex=2, srt=90)
  text(140, 335, 'Class2', cex=2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=3.2, font=2, col='white')
  text(195, 335, res[2], cex=3.2, font=2, col='white')
  text(295, 400, res[3], cex=3.2, font=2, col='white')
  text(295, 335, res[4], cex=3.2, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=2)
  text(30, 85, names(cm$byClass[2]), cex=2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=2)
  text(50, 85, names(cm$byClass[5]), cex=2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=2)
  text(70, 85, names(cm$byClass[6]), cex=2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=2)
  text(90, 85, names(cm$byClass[7]), cex=2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=2.3, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3.2), cex=2)
  text(70, 35, names(cm$overall[2]), cex=2.3, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3.2), cex=2)
}
draw_confusion_matrix(CLR)

# 2.Building of RANDOM FOREST
library(randomForest)
View(train)
RF=randomForest(train$`default payment next month(y)` ~train$`LIMIT_BAL(x1)`+train$`SEX(x2)`+train$`EDUCATION(x3)`+train$`MARRIAGE(x4)`+train$`PAY_0(x6)`+train$`PAY_2(x7)`+train$`PAY_3(x8)`+train$`PAY_4(x9)`+train$`PAY_5(x10)`+train$`PAY_6(x11)`+train$`BILL_AMT1(x12)`+train$`PAY_AMT1(x18)`+train$`PAY_AMT2(x19)`+train$`PAY_AMT3(x20)`+train$`PAY_AMT4(x21)`+train$`PAY_AMT5(x22)`+train$`PAY_AMT6(x23)`, data = train)
RF

#Confusion Matrix
yhatR=as.factor(RF$y)
yR=as.factor(RF$predicted)
CRF=confusionMatrix(data=yhatR,reference = yR)
draw_confusion_matrix(CRF)
CRFt=CRF$table
