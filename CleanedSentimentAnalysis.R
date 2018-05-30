
######### Packages ####
packages <- c('tm', 'SnowballC', 'caTools', 'rpart', 'rpart.plot', 'caret', 'e1071',
              'randomForest','naivebayes', 'purrr', 'tidyr', 'data.table',
              'tidytext', 'dplyr', 'ggplot2', 'igraph', 'ggraph', 'RWeka', 'rJava')
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}

library(sparklyr)
library(tm)
library(SnowballC)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(naivebayes)
library(purrr)
library(tidyr)
library(data.table)
library(tidytext)
library(dplyr)
library(ggplot2)
library(igraph)
library(ggraph)
library(caret)
library(e1071)
library(RWeka)
library(rJava)
library(parallel)
library(snow)
library(doParallel)

######### FUNCTIONS #######

tm_map_cleaner <- function(x) { 
    print("Number Removal")
  x <- tm_map(x, removeNumbers) 
    print("Punct Removal")
  x <- tm_map(x,removePunctuation)
  print("toLOWER")
  x<- tm_map(x, content_transformer(tolower))
  print("Word removal")
  x <- tm_map(x, removeWords, c(stopwords("en"), "br"))
    print("Whitespace")
  x <- tm_map(x, stripWhitespace)
  return(x)
}  

getPredDF <- function(x) { 
  x <- as.data.frame(as.matrix(x))
  (colnames(x) <- make.names(colnames(x)))
  (colnames(x) <- gsub("\\.", " ", colnames(x)))
  x$sentiment <- trainData$sentiment
  return (x)
}

######### Data Manipulation // Data conversions //  ####

## DataImport / Structuring ##
trainData <- fread("labeledTrainData.tsv")
trainData$sentiment <- as.factor(trainData$sentiment == 1)

## Import/Conversion of reviews from DF to VectorSource to corpus
corpus <- VCorpus(VectorSource(trainData$review))

## Sentence preperation for predictions

system.time(corpus <- tm_map_cleaner(corpus))

corpus[[1]]$content

#### DocumentTermMatrix // frequencies  ####
frequencies <- DocumentTermMatrix(corpus, control = list(weighting = weightBin))
frequencies
inspect(frequencies[990:1005, 480:515])

#### Sparse ####
## Sparse Removal 
sparse <- removeSparseTerms(frequencies, 0.97)
sparse
inspect(sparse[990:1005, 480:515])

## DTM to DF conversion. Used for predictions
(reviewsSparse <- getPredDF(sparse))
head(reviewsSparse)
## Train/Test split
set.seed(123)
split <- sample.split(reviewsSparse$sentiment, SplitRatio=0.7) 
## doesnt work with larges matrices with a lot of sparseTerms
trainSparse <- subset(reviewsSparse, split==TRUE)
testSparse <- subset(reviewsSparse, split==FALSE)


### Predictions ####

### Naive Bayes 
classifier = naive_bayes(sentiment ~ ., data=trainSparse)
predicted = predict(classifier, newdata = testSparse)

(NBtable <- table(testSparse$sentiment, predicted))
(NBtable[1,1] + NBtable[2,2]) / nrow(testSparse)


### Sparse 0.99, No weighting 
#         predicted
#        FALSE TRUE
# FALSE  3146  604
# TRUE   1055 2695

# [1] 0.7788

### Sparse 0.99, Weighting = weightTfIdf
#         predicted
#        FALSE TRUE
# FALSE  3031  719
# TRUE    817 2933
# > (NBtable[1,1] + NBtable[2,2]) / nrow(testSparse)
# [1] 0.7952

### Sparse 0.99, Weighting = weightBin
#       predicted
#       FALSE TRUE
# FALSE  3153  597
# TRUE    808 2942
#
# [1] 0.8126667



### Sparse 0.97, No weighting 
#       predicted
#       FALSE TRUE
# FALSE  2880  870
# TRUE    657 3093
# 
# [1] 0.7964

## Sparse 0.97, Weighting = weightTfIdf

#       predicted
#       FALSE TRUE
# FALSE  2815  935
# TRUE    659 3091
# 
# [1] 0.7874667

## Sparse 0.97, Weighting = weightBin
#        predicted
#       FALSE TRUE
#FALSE  2903  847
#TRUE    610 3140
#[1] 0.8057333



### Anomaly ###
#         predicted
#         FALSE TRUE
# FALSE    98 3652
# TRUE      7 3743
# 
# [1] 0.5121333


### Logistic Regression 

classifierLog <- glm(sentiment ~ ., data=trainSparse, family='binomial')

predictLog <- predict(classifierLog, newdata=testSparse, type='response')

(glmtable <- table(testSparse$sentiment, predictLog >= 0.5))
(glmtable[1,1] + glmtable[2,2]) / nrow(testSparse)


### Sparse 0.99, No weighting 
#         FALSE TRUE
# FALSE  3163  587
# TRUE    542 3208
#
# [1] 0.8494667

## Sparse 0.99, weighting = weightTfIdf
# FALSE TRUE
# FALSE  3164  586
# TRUE    531 3219
#
# [1] 0.8510667

## Sparse 0.99, weighting = weightBin
#       FALSE TRUE
# FALSE  3171  579
# TRUE    561 3189
# 
# [1] 0.848



## Sparse 0.97, No weighting 
#       FALSE TRUE
# FALSE  3143  607
# TRUE    608 3142
#
# [1] 0.838

## Sparse 0.97, weighting = weightTfIdf
#       FALSE TRUE
# FALSE  3068  682
# TRUE    557 3193
#
# [1] 0.8348

## Sparse 0.97, weighting = weightBin
#        FALSE TRUE
# FALSE  3057  693
# TRUE    558 3192
# 
# [1] 0.8332

########################################################################################

# ### Supported Vector machine 
# trctrl <- trainControl(method = "boot")
# set.seed(3233)
# 
# cl <- makeCluster(7, type = "SOCK")
# registerDoParallel(cl)
# svm_logic <- train(sentiment ~., data = trainSparse, method = 'vglmAdjCat',
#                     trControl=trctrl,
#                     preProcess = c("center", "scale"),
#                     tuneLength = 10)
# stopCluster(cl)
# predictSVM <- predict(svm_logic, newdata=testSparse)
# 
# (SVMtable <- table(testSparse$sentiment, predictSVM))
# (SVMtable[1,1] + glmtable[2,2]) / nrow(testSparse)


### Sparse 0.99, No weighting 


### Sparse 0.97, No weighting 
#        predictSVM
#        FALSE TRUE
# FALSE  3143  607
# TRUE    608 3142
# 
# [1] 0.838


# PCA = looking for combinations of features that will have more impact on the data set or not. Trick to reduce the number of features but the trick doesnt always help you. 
