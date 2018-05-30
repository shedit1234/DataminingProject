install.packages("ff")
install.packages("ffbase")
install.packages("devtools")
devtools::install_github("edwindj/ffbase2")
install.packages("randomForest")
library(randomForest)
library(ff)
library(ffbase)
library(ffbase2)
trainData <- fread("labeledTrainData.tsv")
unlabeledALLData <- fread("unlabeledTrainData.tsv")
testData <- fread("testData.tsv")

trainData.ffdf <- as.ffdf(factor(trainData), next.rows = 2500)
list <- list(testData, trainData, unlabeledALLData)
library(ff)
library(tm)
library(purrr)
# spark
sc <- spark_connect(master = "local", version = "2.0.0")
system.time(copy_to(sc, trainData, overwrite = TRUE))
system.time(copy_to(sc, unlabeledALLData, overwrite = TRUE))
system.time(copy_to(sc, testData, overwrite = TRUE))

system.time(copy_to(sc, trainSparse, overwrite = TRUE))
system.time(copy_to(sc, testSparse, overwrite = TRUE))
system.time(copy_to(sc, sparse, overwrite = TRUE))

# Imports
train_tbl <- tbl(sc, "trainsparse")
test_tbl <- tbl(sc, "testsparse")


partition <- sparse %>% 
  sdf_partition(train = 0.75, test = 0.25, seed = 8585)


ml_formula <- formula(Negative ~ .)

# Train a logistic regression model
ml_log <- ml_logistic_regression(test_tbl, ml_formula)
## Decision Tree
ml_dt <- ml_decision_tree(train_tbl, ml_formula)

## Random Forest
ml_rf <- ml_random_forest(train_tbl, ml_formula)

## Gradient Boosted Tree
ml_gbt <- ml_gradient_boosted_trees(train_tbl, ml_formula)

## Naive Bayes
ml_nb <- ml_naive_bayes(train_tbl, ml_formula)

## Neural Network
ml_nn <- ml_multilayer_perceptron(train_tbl, ml_formula, layers = c(11,15,2))



ml_models <- list(
  "Logistic" = ml_log,
  "Decision Tree" = ml_dt,
  "Random Forest" = ml_rf,
  "Gradient Boosted Trees" = ml_gbt,
  "Naive Bayes" = ml_nb,
  "Neural Net" = ml_nn
)
ml_nn
Sys.time() - t1
# Create a function for scoring
score_test_data <- function(model, data=test_tbl){
  pred <- sdf_predict(model, data)
  select(pred, Survived, prediction)
}

# Score all the models
ml_score <- lapply(ml_models, score_test_data)

# Function for calculating accuracy
calc_accuracy <- function(data, cutpoint = 0.5){
  data %>% 
    mutate(prediction = if_else(prediction > cutpoint, 1.0, 0.0)) %>%
    ml_classification_eval("prediction", "Survived", "accuracy")
}
