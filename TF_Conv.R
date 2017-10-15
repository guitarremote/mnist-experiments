#Remove exiting Variables and Set Working directory
rm(list=ls())
setwd("C:/Users/Aravind Atreya/Desktop/Kaggle/MNIST")

#Libraries
library(tensorflow)
library(data.table)#Really fast
library(scales)

#Functions

preprocess <- function(input){
  input = as.matrix(input)
  x <- matrix(0, nrow(input), ncol(input)-1)
  y <- matrix(0, nrow(input), 10)
  for(i in 1:nrow(input)){
    x[i,] <- as.matrix(input[i,2:ncol(input)])
    y[i,] <- c(rep(0,input[i,1]),1,rep(0,9-input[i,1]))
  }
  z <- list(a=x,b=y)
  return(z)
}


#Inputs
df_train <- fread("train.csv")
df_test <- fread("test.csv")
df_test <- cbind(label=rep(1,nrow(df_test)),df_test)#Adding a dummy field

z_train <- preprocess(df_train)
train_x <- z_train[[1]]
train_y <- z_train[[2]]

z_test <- preprocess(df_test)
test_x <- z_test[[1]]


train_x <- train_x/255
test_x <- test_x/255

# Create the model
x <- tf$placeholder(tf$float32, shape(NULL, 784L))
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

#y <- tf$nn$softmax(tf$matmul(x, W) + b)

# Define loss and optimizer
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))


#Weight initialization

weight_variable <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape) {
  initial <- tf$constant(0.1, shape=shape)
  tf$Variable(initial)
}

#Convolution and pooling
conv2d <- function(x, W) {
  tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')
}

max_pool_2x2 <- function(x) {
  tf$nn$max_pool(
    x, 
    ksize=c(1L, 2L, 2L, 1L),
    strides=c(1L, 2L, 2L, 1L), 
    padding='SAME')
}

#First Convolutional Layer

W_conv1 <- weight_variable(shape(5L, 5L, 1L, 32L))
b_conv1 <- bias_variable(shape(32L))

x_image <- tf$reshape(x, shape(-1L, 28L, 28L, 1L))
h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 <- max_pool_2x2(h_conv1)

#Second Convolutional layer
W_conv2 <- weight_variable(shape = shape(5L, 5L, 32L, 64L))
b_conv2 <- bias_variable(shape = shape(64L))

h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 <- max_pool_2x2(h_conv2)

#Densely connected layer
W_fc1 <- weight_variable(shape(7L * 7L * 64L, 1024L))
b_fc1 <- bias_variable(shape(1024L))

h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, 7L * 7L * 64L))
h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout

keep_prob <- tf$placeholder(tf$float32)
h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)

#Read out Layer

W_fc2 <- weight_variable(shape(1024L, 10L))
b_fc2 <- bias_variable(shape(10L))

y_conv <- tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)

#Train 
# Create session 
sess <- tf$Session()


cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y_conv), reduction_indices=1L))
train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(tf$global_variables_initializer())

nOfSamples <- 50L
noOfIter <- 20000
pb <- txtProgressBar(min = 0, max = noOfIter, style = 3)
# Train
for (i in 1:noOfIter) {
  setTxtProgressBar(pb, i)
  idx <- sample(c(1:nrow(train_x)), nOfSamples)
  trainSample_x <- train_x[idx,]
  trainSample_y <- train_y[idx,]
  if(i%%1000==0){
  train_accuracy <- accuracy$eval(feed_dict = dict(x = trainSample_x, y_ = trainSample_y,
                                                   keep_prob=1.0),session = sess)
  cat(sprintf("\nStep %d, training accuracy %g\n",i,train_accuracy))
  }
  sess$run(train_step,
           feed_dict = dict(x = trainSample_x, y_ = trainSample_y,keep_prob=0.5))
  
}
close(pb)

# Test trained model on random 50 samples
test_idx <- sample(c(1:nrow(train_x)),1000)
testSample_x <- train_x[test_idx,]
testSample_y <- train_y[test_idx,]


train_accuracy <- accuracy$eval(feed_dict = dict(x = testSample_x, y_ = testSample_y,
                                                 keep_prob=1.0),session = sess)
cat(sprintf("test accuracy %g", train_accuracy))

#Predict for unknown data

predict_results <- sess$run(tf$argmax(y_conv,1L),feed_dict = dict(x=as.matrix(test_x),keep_prob=1.0))

Results_df <- data.frame(ImageId=c(1:nrow(test_x)),Label=predict_results)
write.csv(Results_df,"Sample Submission_TF.csv",row.names = FALSE)

