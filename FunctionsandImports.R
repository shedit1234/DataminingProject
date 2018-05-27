# install.packages("RWeka")
# install.packages("rJava")
# install.packages("tidytext")
# install.packages("qdap")
# install.packages("igraph")
# install.packages("ggraph")
# install.packges("dplyr")
# install.packges("purr")
# install.packges("tidyr")
# install.packges("data.table")
# install.packges("tm")
# install.packges("ggplot2")
#install.packages("sparklyr")


library(sparklyr)
library(purrr)
library(tidyr)
library(data.table)
library(tm)
library(tidytext)
library(dplyr)
library(ggplot2)
#library(qdap)
library(igraph)
library(ggraph)
library(RWeka)
library(rJava)

t1 <- Sys.time()
####### Functions #########

textcleaner <- function(col) {
  col <- col %>% removePunctuation() %>%
    removeNumbers() %>%
    tolower() %>%
    iconv("latin1", "ASCII", sub="") %>% 
    stripWhitespace()
}

dtmconverter <- function(df) { 
  if("doc_id" %in% colnames(df)) {
    print("IF")
    dfsource <- DataframeSource(df)
    print("Source done")
    dfCorpus <- VCorpus(dfsource)
    print("Corpus done")
    dftdm <- TermDocumentMatrix(dfCorpus, control = list(tokenize = BigramTokenizer,
                                                         weighting = weightTfIdf))
    print("tdm done")
    return(dftdm)
  } else {
    print("ELSE")
    df <- df[,c(1, 3, 2)]
    colnames(df) <- c("doc_id", "text", "sentiment")
    dfsource <- DataframeSource(df)
    print("Source done")
    dfCorpus <- VCorpus(dfsource)
    print("Corpus done")
    dftdm <- TermDocumentMatrix(dfCorpus, control = list(tokenize = BigramTokenizer, 
                                                         weighting = weightTfIdf))
    print("tdm done")
    return(dftdm)
  }
}

barGraphAnalysis <- function(df, tokenstr, number = NULL) {
  
  df <- tokenize(df, tokenstr, number)
  
  df <- df %>% 
    count(words, sentiment, sort = TRUE) %>%
    ungroup()
  
  print(df)
  
  df <- df %>% 
    group_by(sentiment) %>%
    top_n(10) %>%
    ungroup() %>%
    mutate(words = reorder(words, n)) %>%
    ggplot(aes(words, n, fill = sentiment)) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~sentiment, scales = "free_y") +
    labs(y = "Contribution to sentiment",
         x = NULL) +
    coord_flip()
  print(df)
}

bigramRelationPlot <- function(df) {
  ## igraph, dplyr and ggraph
  df <- tokenize(df, "ngrams", 2) %>% 
        select(words, sentiment) 
  dfcount <- df %>% 
    separate(words, c("word1", "word2")) %>% 
    select(word1, word2, sentiment) %>%
    count(word1, word2, sort = TRUE)
  
  dfgraph <- dfcount %>%
    filter(n > 200) %>%
    graph_from_data_frame()
  
  set.seed(2017)
  
  a <- grid::arrow(type = "closed", length = unit(.15, "inches"))
  
  ggraph(dfgraph, layout = "fr") +
    geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                   arrow = a, end_cap = circle(.07, 'inches')) +
    geom_node_point(color = "lightblue", size = 5) +
    geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
    theme_void()
  
}

tokenize <- function(df, tokenstr,  number = NULL){ 
  if (is.null(number)){
    df <- unnest_tokens(df, words, review, token = tokenstr)
  } 
  else { df <- unnest_tokens(df, words, review, token = tokenstr, n = number)
  }
  return(df)
} 

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

#####Imports########

trainData <- fread("labeledTrainData.tsv")
unlabeledALLData <- fread("unlabeledTrainData.tsv")
testData <- fread("testData.tsv")


## General cleaning

trainData$review <- textcleaner(trainData$review)
#trainData$review <- removeWords(trainData$review, words = c(stopwords("en")))
trainData$sentiment <- as.logical(trainData$sentiment)

## testsplit 
trainData.train <- trainData[1:round(nrow(trainData)*0.75)]
trainData.test <- trainData[round(nrow(trainData)*0.75+1):nrow(trainData)]







## spark 
# sc <- spark_connect(master = "local", version = "2.0.0")
#   system.time(copy_to(sc, trainData, overwrite = TRUE))
# # Imports 
# trainData <- tbl(sc, "trainData")


## DocumentMatrix conversions 
trainData.DTM.test <- dtmconverter(trainData.test)
trainData.DTM.test <- removeSparseTerms(trainData.DTM.test, sparse = 0.99)


trainData.Matrix.test <- as.matrix(trainData.DTM.test)



## Random stuff not neccesary I think 

# trainData.DTM.train <- dtmconverter(trainData.train)
# trainData.Matrix.train <- as.matrix(trainData.DTM.train)
# 
# 
# sort(rowSums(as.matrix(trainData.DTM)))

# #Negative reviews
# trainDataN.TDM <- tdmconverter(trainDataN)
# trainDataN.TDM <- removeSparseTerms(trainDataN.TDM, sparse = 0.95)
# trainDataN.Matrix <- as.matrix(trainDataN.TDM)
# Negativeterm_frequency<- sort(rowSums(trainDataN.Matrix), decreasing= TRUE)
# 
# barplot(Negativeterm_frequency[c(1:30)], col = "tan", las = 2)
# #positive reviews
# 
# trainDataP.TDM <- tdmconverter(trainDataP)
# 
# trainDataMatrix <- as.matrix(trainData.TDM)





# vocab <-  trainData %>% 
#           tokenize("ngrams", n =2) %>% 
#           select(words, sentiment) 
#           
#  
# vocab_counts <- vocab %>% 
#                 separate(words, c("word1", "word2")) %>% 
#                 select(word1, word2, sentiment) %>%
#                 count(word1, word2, sort = TRUE)
# vocab_counts
# ##########################################################################
# 

