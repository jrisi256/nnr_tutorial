
# For this tutorial we'll classify audio data by emotions. I chose audio data because it hits three birds
# with one stone: Classifying audio data is the same as classifying images, which is the same
# asclassifying any vector of numbers.

library(neuralnet)
library(MLmetrics)

########################
## Introduction and Data
########################

# imager is used for importing and manipulating data.
# It can take a while to install. If you don't want to
# install it you can skip this section. The data will be
# read in as a CSV later.

library(imager)

# This is the data source. It's from the Toronto Emotional Speech Set. 
# The set consists of both old adult female (OAF) and young adult female (YAF) 
# voice samples saying various words with specific emotions. 
# We'll work with the happy and angry samples

# For classification, audio is converted to an image of a mel spectrogram. 

# load a sample image with imager
sample <- load.image('happy/H_OAF_back.png')
plot(sample)

# The image is actually just a vector of numeric pixel values.
# here are the first 10 pixel values
sample[1:10]

# Each pixel is a feature. In order to reduce the features in
# our data set we can resize the image to reduce the pixel count.
sample <- resize(sample, 32, 32)
plot(sample)

# Now we will resize all images in the data set and add their
# pixel values to a data frame

# set the desired dimensions we will convert the images to
dim1 <- 32
dim2 <- 32

# list images in the happy directory
happy <- list.files('happy', full.names=T)
# instantiate the data frame
happy_df <- data.frame()
# generate column names for the data frame based on the number of pixels in the image
# these pimages will be 32*32 = 1024 pixels. So i create columns named px1......px1024
col_names <- sapply('px', paste, seq(dim1*dim2), sep='')
# load each image, resize it, and then add the vector to the data frame
for(file in happy){
    img <- load.image(file)
    img <- resize(img, dim1, dim2)
    happy_df <- rbind(happy_df, img)
}
# set the column names
colnames(happy_df) <- col_names
# add a label to the data. Happy will be 1 and angry 0
happy_df$label <- 1


# Now repeat the above for the angry spectrograms
angry <- list.files('angry', full.names=T)
# instantiate the data frame
angry_df <- data.frame()
# generate column names
col_names <- sapply('px', paste, seq(dim1*dim2), sep='')
# load each image, resize it, and then add the vector to the data frame
for(file in angry){
    img <- load.image(file)
    img <- resize(img, dim1, dim2)
    angry_df <- rbind(angry_df, img)
}
# set the column names
colnames(angry_df) <- col_names
# add a label to the data
angry_df$label <- 0

# combine both data sets
df <- rbind(happy_df, angry_df)
# save it to a csv
write.csv(df, 'pixels.csv')


###########################
## Classifying with neuralnet
###########################

# import the data from the csv
df <- read.csv('pixels.csv')
# this will randomize the order of the data
set.seed(123)
df <- df[sample(1:nrow(df)),]

# now divide the data in to training and test sets. I use an 80:20 split
train <- df[1:592,]
test <- df[592:740,]


# define the formula
f <- as.formula(paste('label ~', paste(colnames(df[,1:1024]), collapse = " + ")))


# Fit our first model
# train the network
set.seed(123)
nn <- neuralnet(f, 
                data=train,
                stepmax = 50, # the maximum number of training steps. Setting it at 50 for this model to limit run time
                learningrate.factor = list(minus = 1, plus = 2), # try adjusting the learning rate if your model doesn't converge
                hidden = c(128, 32, 8), # The hidden layers a number of neurons in each
                linear.output = FALSE, # The type of output. True for regression tasks, False for classification
                algorithm = 'rprop+', # The algorith for calculating the network
                act.fct = 'logistic' # The activation function used between neurons                
                )

# This model failoed to converge. This can often happen if your network is too large/small or if the learning rate
# is too large. Let's try simplifying the network and turning down the learning rate

# train the network
set.seed(123)
nn <- neuralnet(f, 
                data=train,
                stepmax = 1e+05, # the maximum number of training steps
                learningrate.factor = list(minus = 0.5, plus = 1.2), # try adjusting the learning rate if your model doesn't converge
                hidden = c(128, 32, 8), # The hidden layers a number of neurons in each
                linear.output = FALSE, # The type of output. True for regression tasks, False for classification
                algorithm = 'rprop+', # The algorith for calculating the network
                act.fct = 'logistic' # The activation function used between neurons                
                )

# this provides a quick view of the training results
head(nn$result.matrix)

# Now lets see how the model preforms by making predictions on our test set. 
# The model will assign a continuous value rather than a binary classification for each observation.

# drop the label from the test set so the network can predict it
nn_test <- subset(test, select=-c(label))

# predict labels for the test set
results <- compute(nn, nn_test)

# Create a data frame of true labels and results
r_df <- data.frame(actual=test$label, prob = results$net.result)

# If the predicted value is over, .5 label happy happy, otherwise angry
r_df$pred <- ifelse(r_df$prob > 0.5, 1, 0)
# flag correct predictions
r_df$correct <- ifelse(r_df$pred == r_df$actual, 1, 0)
# calculate the accuracy of predictions
sum(r_df$correct)/nrow(r_df)

# print the confusion matrix as well as preformance metrics
ConfusionMatrix(y_pred= r_df$pred, y_true = r_df$actual)

AUC(y_pred= r_df$pred, y_true = r_df$actual)
PRAUC(y_pred= r_df$prob, y_true = r_df$actual)

# this model fit pretty well. For the sake of demonstration, lets try fitting a simpler model with a single hidden layer
# define the formula
f <- as.formula(paste('label ~', paste(colnames(df[,1:1024]), collapse = " + ")))

# train the network
set.seed(123)
nn <- neuralnet(f, 
                data=train,
                stepmax = 1e+05, # the maximum number of training steps
                learningrate.factor = list(minus = 0.5, plus = 1.2), # try adjusting the learning rate if your model doesn't converge
                hidden = c(64), # The hidden layers a number of neurons in each
                linear.output = FALSE, # The type of output. True for regression tasks, False for classification
                algorithm = 'rprop+', # The algorith for calculating the network
                act.fct = 'logistic' # The activation function used between neurons                
                )

# this provides a quick view of the training results
head(nn$result.matrix)

# drop the label from the test set so the network can predict it
nn_test <- subset(test, select=-c(label))

# predict labels for the test set
results <- predict(nn, nn_test)

# Create a data frame of true labels and results
r_df <- data.frame(actual=test$label, prob = results)

# If the predicted value is over, .5 label happy happy, otherwise angry
r_df$pred <- ifelse(r_df$prob > 0.5, 1, 0)
# flag correct predictions
r_df$correct <- ifelse(r_df$pred == r_df$actual, 1, 0)
# calculate the accuracy of predictions
sum(r_df$correct)/nrow(r_df)

ConfusionMatrix(y_pred= r_df$pred, y_true = r_df$actual)

AUC(y_pred= r_df$pred, y_true = r_df$actual)
PRAUC(y_pred= r_df$prob, y_true = r_df$actual)

#This model failed to learn anything. Effective training is about finding the right balance of complexity, 
# learning rate, and training times.

###########################
## Tensorflow with Keras
###########################

# Run these commands in a console in order to install Tensorflow with Keras
#install.packages('tensorflow')
#library(tensorflow)
#install_tensorflow()

#install.packages('keras')
#library(tensorflow)
#install_tensorflow()

# once installed, we can just import keras and everything it needs from
# tensorflow will come with it.
library(keras)
# dplyr for the pipe operator
library(dplyr)

# define the image size we want to use
img_size <- c(32, 32)

# define the batch size. This is the number of observations passed through the network before model parameters are updated
# Larger or smaller batch sizes will produce better/worse results depending on the model. 32 is a standard starting point.
# If your computer runs out of memory while training the model, try lowering the batch size.
batch_size <- 32


# Data Generator for Training Set. Keras infers the labels for the data based on the directory
train_image_array_gen <- flow_images_from_directory(directory = paste0(getwd(),"/keras/training/"),
                           target_size = img_size, 
                           color_mode = "grayscale", 
                           batch_size = batch_size,
                           seed = 123)

# Data Generator for Validation Set
validation_image_array_gen <- flow_images_from_directory(directory = paste0(getwd(),"/keras/test/"),
                           target_size = img_size, 
                           color_mode = "grayscale", 
                           batch_size = batch_size,
                           seed = 123)

# The number of files used by our training generator.
# We'll call these values later when training
n_train <- train_image_array_gen$n
# The number of files used by our validation generator
n_test <- validation_image_array_gen$n
# the number of categories in our data
n_cat <- 2

# We're going to create the same model we used in neuralnet, a feed forward network
# with fully connected (dense) layers and a logistic activation function.
# In keras a feed foward model is called a sequential model
# Define the type of model. Sequential is a linear feed forward model
model <- keras_model_sequential()

# Now we define each layer of the model
model %>%
    # The first layer needs to define the dimensions of the input data. In this case, a 32x32 matrix of pixel values
    layer_flatten(input_shape = c(32,32)) %>%
    # Now we add our dense layers. Units is the number of neurons and activation is the function used to determine
    # if neurons on the next layer are activated. The last layer uses a softmax function to return a value between
    # zero and one.
    layer_dense(units = 128, activation = 'sigmoid') %>%
    layer_dense(units = 16, activation = 'sigmoid') %>%
    layer_dense(units = 8, activation = 'sigmoid') %>%
    layer_dense(units = 2, activation = 'softmax')

# Compile the model. In this stage we define how to measure model preformance
# as well as the function we will use to adjust model parameters during training.
model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = 0.001),
    metrics = "accuracy"
)

# print the model
model

# Now we can fit the model. epochs is the number of complete passes through the data
# that will occur during training
keras_nn <- model %>% 
    fit_generator(
    # Pass our training data generator
    train_image_array_gen,
  
    # Define the number of epochs as well as the steps per epoch.
    # since we want an epoch to be one complete pass through of the data, we set the
    # number of steps equal to the total number of samples divided by the batch size
    steps_per_epoch = as.integer(n_train/batch_size), 
    epochs = 10, 
  
    # Now we add the validation data
    validation_data = validation_image_array_gen,
    validation_steps = as.integer(n_test/batch_size),
  
    # print progress
    verbose = 2
)

# Now if we call plot on the model we can see its preformance
# and if we just call the model itself it will return evaluation metrics
plot(keras_nn)
keras_nn

# This model wasn't very successful, so let's try lowering the learning rate
# and adding more neurons to the network.


# Define the type of model. Sequential is a linear feed forward model
model <- keras_model_sequential()

# Now we define each layer of the model
model %>%
    # The first layer needs to define the dimensions of the input data. In this case, a 64x64 matrix of pixel values
    layer_flatten(input_shape = c(32,32)) %>%
    # Now we add our dense layers. Units is the number of neurons and activation is the function used to determine
    # if neurons on the next layer are activated.
    layer_dense(units = 512, activation = 'sigmoid') %>%
    layer_dense(units = 128, activation = 'sigmoid') %>%
    layer_dense(units = 16, activation = 'sigmoid') %>%
    layer_dense(units = 2, activation = 'softmax')

# Compile the model
model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = 0.0001),
    metrics = "accuracy"
)

# print the model
model

keras_nn2 <- model %>% 
    fit_generator(
    # Pass our training data generator
    train_image_array_gen,
  
    # Define the number of epochs as well as the steps per epoch.
    # since we want an epoch to be one complete pass through of the data, we set the
    # number of steps equal to the total number of samples divided by the batch size
    steps_per_epoch = as.integer(n_train/batch_size), 
    epochs = 15, 
  
    # Now we add the validation data
    validation_data = validation_image_array_gen,
    validation_steps = as.integer(n_test/batch_size),
  
    # print progress
    verbose = 2
)

plot(keras_nn2)
keras_nn2
