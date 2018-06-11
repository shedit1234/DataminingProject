# Prediction by classificationn with sparseTerms 0.99 and TfIdf weighting Acc: 0.62
source("/Users/Shedit/Downloads/06-DATA-ANALYSIS-AND-MINING-part2-semester/Individual assignment Data mining/DataminingMovies/FunctionsandImports.R")
library(naivebayes)
#### Graphs #########

#barGraphAnalysis(trainData, "ngrams", n = 2)

bigramRelationPlot(trainData)

#### Prediction ####

classifier = naive_bayes(t(trainData.Matrix.test[,1:5000]), trainData.test$sentiment[1:5000])

#classifier2 = randomForest(t(trainData.Matrix.test[,1:5000]), factor(trainData.test$sentiment[1:5000]))
predicted = predict(classifier, t(trainData.Matrix.test[,5001:6250]))

#trainData[5001:6250, -3]
#head(predicted, n = 20)

Summaretable <- data.table(id = trainData[5001:5021,1], train = trainData[5001:5025,2], pred = head(predicted, n = 20))
Summaretable

Confmatrix <- table(as.matrix(trainData.test[5001:6250,2]), predicted)
Confmatrix

Accuracy <- (Confmatrix[1,1] + Confmatrix[2,2]) / sum(Confmatrix)
Accuracy

Sys.time() - t1