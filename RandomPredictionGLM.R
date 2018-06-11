
### Anomaly ###
#         predicted
#         FALSE TRUE
# FALSE    98 3652
# TRUE      7 3743
# 
# [1] 0.5121333



### Logistic Regression 

classifierLog <- glm(sentiment ~ ., data=trainSparse, family='binomial')

(perf <- glance(classifierLog))

# Calculate pseudo-R-squared
(pseudoR2 <- 1 - perf$deviance/perf$null.deviance)

testSparse$pred <- predict(classifierLog, newdata=testSparse, type='response')

GainCurvePlot(testSparse, "pred", "sentiment", "sparrow survival model")
(glmtable <- table(testSparse$sentiment, testSparse$pred >= 0.5))
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