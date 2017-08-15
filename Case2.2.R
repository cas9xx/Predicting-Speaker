setwd("~/Desktop/Fall 2016 Semester/SYS 6018/Case Study 2")
library(tm)

debate <- read.csv("debate.csv", stringsAsFactors = F)

# selecting who to look at
whoToLookAt <- c('Clinton', 'Trump', 'Cooper', 'Holt', 'Quijano', 'Raddatz', 'QUESTION')

debate <- debate[debate$Speaker %in% whoToLookAt, ]
debate$Text <- gsub('[^A-Za-z ]', '', debate$Text)

debate$Speaker[!(debate$Speaker %in% c('Clinton', 'Trump'))] <- "MODERATOR"

# for the purpose of this example, we only care about content.
document.data.frame = as.data.frame(debate[,"Text"], stringsAsFactors = FALSE)

# convert this part of the data frame to a corpus object.
debateCorpus = VCorpus(DataframeSource(document.data.frame))

# regular indexing returns a sub-corpus
inspect(debateCorpus[1:2])

# double indexing accesses actual documents
debateCorpus[[1]]
debateCorpus[[1]]$content

# set sparsity bounds
lb <- round(length(debateCorpus)*.05, 0)
ub <- round(length(debateCorpus)*.99, 0)

# compute TF-IDF matrix and inspect sparsity
debateCorpus.tfidf = DocumentTermMatrix(debateCorpus, 
                                        control = list(weighting = weightTfIdf,
                                                       bounds = list(global = c(lb, ub))))
debateCorpus.tfidf

# inspect sub-matrix:  first 5 documents and first 5 terms
as.matrix(debateCorpus.tfidf[1:5,1:5])
debateCorpus.tfidf[1:5,1:5]


##### Reducing Term Sparsity #####

# there's a lot in the documents that we don't care about. clean up the corpus.
debateCorpus.clean = tm_map(debateCorpus, stripWhitespace)                          # remove extra whitespace
debateCorpus.clean = tm_map(debateCorpus.clean, removeNumbers)                      # remove numbers
debateCorpus.clean = tm_map(debateCorpus.clean, removePunctuation)                  # remove punctuation
debateCorpus.clean = tm_map(debateCorpus.clean, content_transformer(tolower))       # ignore case
debateCorpus.clean = tm_map(debateCorpus.clean, removeWords, stopwords("english"))  # remove stop words
debateCorpus.clean = tm_map(debateCorpus.clean, stemDocument)                       # stem all words

# compare original content of document 1 with cleaned content
debateCorpus[[1]]$content
debateCorpus.clean[[1]]$content

# recompute TF-IDF matrix
debateCorpus.clean.tfidf = DocumentTermMatrix(debateCorpus.clean, 
                                              control = list(weighting = weightTfIdf,
                                                             bounds = list(global = c(lb, ub))))

# reinspect the first 5 documents and first 5 terms
debateCorpus.clean.tfidf
as.matrix(debateCorpus.clean.tfidf[1:5,1:5])

########### CREATE DATA FRAME TO MODEL FROM
moddf <- data.frame(as.matrix(debateCorpus.clean.tfidf))
moddf$y <- as.factor(debate$Speaker)

####### MODELS
#Mulitnomial Logistic Regression 
library(VGAM)
library(glmnet)
moddf <- moddf[325:404, ]
moddf$y <- as.factor(as.character(moddf$y))
mod2 <- vglm(moddf$y ~ ., family=multinomial(refLevel=1), data=moddf)


####Cross Validated Multinomial Regression model 
#moddf$y= (as.numeric(as.factor(moddf$y)))
#modm= (as.matrix(moddf))
#cv.glmnet(modm[,1:60], modm[,61], nfolds= 5)


#View(predict(mod2, type = 'response'))
moddf <- cbind(moddf, predict(mod2, type = 'response'))
#View(moddf[, (ncol(moddf)-10):ncol(moddf)])

# create an index
#moddf$index <- as.numeric(row.names(moddf))
#moddf <- moddf[order(moddf$preds1), ]
#moddf$probOrder <- seq(1,nrow(moddf))

#moddf$rowSums <- rowSums(subset(moddf, select = -c(y,preds1, index, probOrder)))

#View(moddf[moddf$rowSums==0, c('y', 'preds1')])
#moddf['accuracy']= NA

moddf['prediction']= NA
for (i in 1:length(moddf$y)){
  moddf[i,"prediction"]= names(which.max(moddf[i,c('Clinton', "MODERATOR", 'Trump')]))}
moddf= moddf[,-c(62:64)]
moddf$prediction <- as.factor(moddf$prediction)


#sum(moddf$y!= moddf$prediction)/length(moddf$y) #0.2368046

library(nnet)
mod3 <- multinom(y~., data=moddf)
preds3 <- predict(mod3, type="probs")

#LogLoss
MultiLogLoss(y_pred=preds3, y_true=moddf$y)
#0.4897276

categHat <- levels(moddf$y)[max.col(preds3)]

facHat <- factor(categHat, levels=levels(moddf$y))
cTab   <- xtabs(~ y + facHat, data=moddf)
addmargins(cTab)

#Classification Rate
(CCR <- sum(diag(cTab)) / sum(cTab))
#0.7631954

theCV <- function(train, k=5, returnDF=F) {
  #make column for preds
  train$preds <- factor(x=rep(levels(train$y)[1], nrow(train)), 
                        levels = levels(train$y))
  #randomize the indexes
  nums <- sample(row.names(train), nrow(train))
  #split the indexes into k groups
  nv <- split(nums, cut(seq_along(nums), k, labels = FALSE))
  #subset the training data into k folds
  trainlist <- list()
  for (i in 1:k) {
    trainlist[[i]] <- train[nv[[i]], ]
  }
  #trainlist
  #run on each fold
  for (i in 1:k) {
    ftrainlist <- trainlist[-i]
    ftrain <- ftrainlist[[1]]
    for (j in 2:length(ftrainlist)) {
      ftrain <- rbind(ftrain, ftrainlist[[j]])
    }
    ############# THE MODEL #######################
    #mod <- lm(as.formula(paste(form,' - preds')), data = ftrain) ### the model
    mod <- multinom(y~., data=ftrain) ### the model
    ###############################################
    trainlist[[i]]$preds <- predict(mod, type= "probs",newdata = trainlist[[i]])
    print(paste("finished fold", i))
  }
  #reassemble
  cvdata <- ftrainlist[[1]]
  for (j in 2:length(trainlist)) {
    cvdata <- rbind(cvdata, trainlist[[j]])
  }
  # return stats
  ##raw accuracy
  ra <- nrow(cvdata[cvdata$y == cvdata$preds,]) / nrow(cvdata)
  print(paste("Raw Accuracy:", ra))
  ##balanced error rate
  ###http://spokenlanguageprocessing.blogspot.com/2011/12/evaluating-multi-class-classification.html
  ###http://www2.cmp.uea.ac.uk/~sjc/read-cox-Interspeech-07.pdf
  nk <- length(levels(train$y))
  recall <- numeric(nk)
  for (i in 1:nk) {
    ck <- levels(train$y)[i]
    recall[i] <- nrow(cvdata[cvdata$y==ck & cvdata$preds==ck,]) / nrow(cvdata[cvdata$y==ck,])
  }
  BER <- 1 - (sum(recall)/nk)
  print(paste("Balanced Error Rate:", BER))
  # return actual predictions
  cvdata <- cvdata[order(as.numeric(row.names(cvdata))), ]
  if(returnDF == T) {
    return(cvdata[,c('y', 'preds')])
  } else {
    return()
  }
}

# theCV <- function(train, k=5, returnDF=F) {
#   #make column for preds
#   train$preds <- factor(x=rep(levels(train$y)[1], nrow(train)), 
#                         levels = levels(train$y))
#   #randomize the indexes
#   nums <- sample(row.names(train), nrow(train))
#   #split the indexes into k groups
#   nv <- split(nums, cut(seq_along(nums), k, labels = FALSE))
#   #subset the training data into k folds
#   trainlist <- list()
#   for (i in 1:k) {
#     trainlist[[i]] <- train[nv[[i]], ]
#   }
#   #trainlist
#   #run on each fold
#   for (i in 1:k) {
#     ftrainlist <- trainlist[-i]
#     ftrain <- ftrainlist[[1]]
#     for (j in 2:length(ftrainlist)) {
#       ftrain <- rbind(ftrain, ftrainlist[[j]])
#     }
#     ############# THE MODEL #######################
#     #mod <- lm(as.formula(paste(form,' - preds')), data = ftrain) ### the model
#     # k <- train$y
#     # train$y <- as.numeric(train$y)
#     mod <- vglm(train$y ~ ., family=multinomial(refLevel=1), data=train) ### the model
#     # train$y <- k
#     ###############################################
#     #View(predict(mod2, type = 'response'))
#     train <- cbind(train, predict(mod, type = 'response'))
#     train['prediction']= NA
#     for (i in 1:length(mtrain$y)){
#       train[i,"prediction"]= names(which.max(train[i,c('Clinton', "MODERATOR", 'Trump')]))}
#     train= train[,-c(62:64)]
#     train$prediction <- as.factor(train$prediction)
#     
#     preds <- train$prediction
#     
#     trainlist[[i]]$prediction <- preds
#     print(paste("finished fold", i))
#   }
#   #reassemble
#   cvdata <- ftrainlist[[1]]
#   for (j in 2:length(trainlist)) {
#     cvdata <- rbind(cvdata, trainlist[[j]])
#   }
#   # return stats
#   ##raw accuracy
#   ra <- nrow(cvdata[cvdata$y == cvdata$preds,]) / nrow(cvdata)
#   print(paste("Raw Accuracy:", ra))
#   ##balanced error rate
#   ###http://spokenlanguageprocessing.blogspot.com/2011/12/evaluating-multi-class-classification.html
#   ###http://www2.cmp.uea.ac.uk/~sjc/read-cox-Interspeech-07.pdf
#   nk <- length(levels(train$y))
#   recall <- numeric(nk)
#   for (i in 1:nk) {
#     ck <- levels(train$y)[i]
#     recall[i] <- nrow(cvdata[cvdata$y==ck & cvdata$preds==ck,]) / nrow(cvdata[cvdata$y==ck,])
#   }
#   BER <- 1 - (sum(recall)/nk)
#   print(paste("Balanced Error Rate:", BER))
#   # return actual predictions
#   cvdata <- cvdata[order(as.numeric(row.names(cvdata))), ]
#   if(returnDF == T) {
#     return(cvdata[,c('y', 'preds')])
#   } else {
#     return()
#   }
# }


theCV(moddf)












# ##### PLOT
# library(ggplot2)
# # un-ordered
# ggplot(moddf[moddf$rowSums> 0, ], aes(x=index, y=preds1, colour = y)) + geom_point()
# 
# # ordered by probability
# ggplot(moddf[moddf$rowSums> 0, ], aes(x=probOrder, y=preds1, colour = y)) + geom_point()
