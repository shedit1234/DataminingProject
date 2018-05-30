packages <- c('tm', 'SnowballC', 'caTools', 'rpart', 'rpart.plot', 'randomForest')
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}
library(tm)
library(SnowballC)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(naivebayes)

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

# Download dataset, if it does not exist.
trainData$sentiment <- as.factor(trainData$sentiment == 1)

table(trainData$sentiment)

corpus <- VCorpus(VectorSource(trainData$review))
corpus[[1]]

# Convert to lower-case.
corpus <- tm_map(corpus, content_transformer(tolower))
#corpus <- tm_map(corpus, PlainTextDocument)
corpus[[1]]

corpus <- tm_map(corpus, removePunctuation)
corpus[[1]]

corpus <- tm_map(corpus, removeWords, c(stopwords('english')))
corpus[[1]]

corpus <- tm_map(corpus, stripWhitespace)
corpus[[1]]$content

corpus <- tm_map(corpus, trimws)
corpus[[1]]

frequencies <- DocumentTermMatrix(corpus)
inspect(frequencies[990:1005, 480:515])

# Only 56 terms appear 20 or less times in our tweets.
findFreqTerms(frequencies, lowfreq = 20)

# Remove these words that are not used very often. Keep terms that appear in 0.5% or more of tweets.
# Only 309 terms now (out of previous 3289 terms).

sparse <- removeSparseTerms(frequencies, 0.98)
inspect(sparse[990:1005, 480:515])

reviewsSparse <- as.data.frame(as.matrix(sparse))
(colnames(reviewsSparse) <- make.names(colnames(reviewsSparse)))
(colnames(reviewsSparse) <- gsub("\\.", " ", colnames(reviewsSparse)))
reviewsSparse$sentiment <- trainData$sentiment

# Build a training and testing set.
set.seed(123)
split <- sample.split(reviewsSparse$sentiment, SplitRatio=0.7)
trainSparse <- subset(reviewsSparse, split==TRUE)
testSparse <- subset(reviewsSparse, split==FALSE)

# Build a CART classification regression tree model on the training set.
modelCART <- rpart(sentiment ~ ., data=trainSparse, method='class')
prp(modelCART)

predictCART <- predict(modelCART, newdata=testSparse, type='class')
(rparttable <- table(testSparse$sentiment, predictCART))

#    predictCART
#        FALSE TRUE
# FALSE   2043 1707
# TRUE    534 3216
#Accuracy 0.7012

(rparttable[1,1]+rparttable[2,2]) / nrow(testSparse)

# Try a random forest model.
set.seed(123)
tweetRF <- randomForest(sentiment ~ ., data = trainSparse, ntree = 100, do.trace = TRUE)

predictRF <- predict(tweetRF, newdata=testSparse)
table(testSparse$Negative, predictRF)
#         FALSE TRUE
# FALSE   293    7
# TRUE     33   22
# Accuracy: 0.8873239
(11529 + 3035) / nrow(testSparse)

# Build a logistic regression model.
tweetLog <- glm(sentiment ~ ., data=trainSparse, family='binomial')
summary(tweetLog)
predictLog <- predict(tweetLog, newdata=testSparse, type='response')
(glmtable <- table(testSparse$sentiment, predictLog >= 0.5))
#         FALSE TRUE
# FALSE  2988  762
# TRUE    428 3322
# Accuracy: 0.8413333
(glmtable[1,1] + glmtable[2,2]) / nrow(testSparse)


library(naivebayes)
classifier = naive_bayes(sentiment ~ ., data=trainSparse, laplace = 0.00001)

predicted = predict(classifier, newdata = testSparse, type = 'prob')

(NBtable <- table(testSparse$sentiment, predicted))
#         FALSE TRUE
# FALSE  2936  814
# TRUE    683 3067
# Accuracy: 0.8344
(NBtable[1,1] + NBtable[2,2]) / nrow(testSparse)

install.packages("pROC")
library(pROC)
plot(roc(testSparse$sentiment, predictLog),
     col="red", lwd=3, main="ROC-curve")
auc(roc(testSparse$sentiment, predictLog))
## 
## Call:
## roc.default(response = test_set$bad_widget, predictor = glm_response_scores,     direction = "<")
## 
## Data: glm_response_scores in 59 controls (test_set$bad_widget FALSE) < 66 cases (test_set$bad_widget TRUE).
## Area under the curve: 0.9037
glm_simple_roc <- simple_roc(testSparse$sentiment=="TRUE", predictLog)
with(glm_simple_roc, points(1 - FPR, TPR, col=1 + labels))
