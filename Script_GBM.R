#Remove existing variables
rm(list = ls())


#Load libraries
library(gbm)
library(caret)
library(neuralnet)
library(h2o)

#Function for converting results back to numbers
to_Number <-  function(x){
  if (x=="X0"){x <- 0}
  else if (x=="X1"){x <- 1}
  else if (x=="X2"){x <- 2}
  else if (x=="X3"){x <- 3}
  else if (x=="X4"){x <- 4}
  else if (x=="X5"){x <- 5}
  else if (x=="X6"){x <- 6}
  else if (x=="X7"){x <- 7}
  else if (x=="X8"){x <- 8}
  else {x <- 9}
}

#Set wroking directory
setwd("C:/Users/Aravind Atreya/Desktop/Kaggle/MNIST")

#Inputs
train <- read.csv("train.csv")
test <- read.csv("test.csv")

labels <- as.factor(train[,1])
table(labels)
train$label <- NULL

#Checking out the dimesnsions of our data frames
dim(train)
dim(test)

#PCA
#Some pixels have zero value in all images.They might as well be dropped
#Doubt: Pixels to be dropped are different in train and test, though most of them are same.
#So how correct is this???

number_zeros_train <- sapply(train,function(x){length(which(x==0))})
num_zeros_test <- sapply(test,function(x){length(which(x==0))})
drop_pixels_train <- which(number_zeros_train==dim(train)[1])
drop_pixels_test <- which(num_zeros_test==nrow(test))
train_new <- train[,-drop_pixels_train]
test_new <- test[,-drop_pixels_train]


#Let's take a look at our new train set after dropping unnecessary columns
dim(train_new)

#Apply PCA on train_new
prin_comp_train <- prcomp(train_new,scale. = T)
test_pca <- predict(prin_comp_train,test_new)

#Standard Deviation
std_dev_train <- prin_comp_train$sdev

#Variance
pr_var_train <- std_dev_train^2

#Proportion of variance explained by each Principal component
prop_varex_train <- pr_var_train/sum(pr_var_train)


#Plot
plot(prop_varex_train,xlab = "Principal Components",
     ylab="Percentage of Variance Explained",
     main = "PCA")

#Scree PLot
plot(cumsum(prop_varex_train),xlab = "Principal Component",
     ylab="Cumulative percentage of Variance Explained",
     main="Cumulative PCA")

#From plots we can see that 500 principal components explain 99% of thee variance
sum(prop_varex_train[1:500])

#PCA Train
train_pca <- data.frame(prin_comp_train$x)[1:500]
test_pca <- test_pca[,c(1:500)]
dim(train_pca)
dim(test_pca)

#Before runing the model, we need to convert the numbers to syntactically valid names
train_pca$Label <-make.names(labels)

#Using Caret
#This took near 7 hours and still didn't finish had to stop it :(

fitControl <- trainControl(method="cv",number = 5,classProbs = TRUE)

set.seed(5)

gbm_fit <- train(Label~.,method="gbm",trControl=fitControl,metric="Accuracy",data=train_pca)

gbm_pred <- predict(gbm_fit,test_pca,type="raw")

#Convert results to numbers for submissions
gbm_pred_number <- sapply(gbm_pred,to_Number)

#Generate submission file
Results <- data.frame(ImageID=c(1:length(gbm_pred_number)),Label=gbm_pred_number)
write.csv(Results,"Submission_GBM.csv",row.names = F)





