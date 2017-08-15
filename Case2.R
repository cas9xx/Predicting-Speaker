setwd("~/Desktop/Fall 2016 Semester/SYS 6018/Case Study 2")
library(tm)
library(SnowballC)
library(ggplot2)

debate <- read.csv("debate.csv", stringsAsFactors = F)
debate <- debate[debate$Speaker %in% c('Clinton', 'Trump'), ]
debate$Text <- gsub('[^A-Za-z ]', '', debate$Text)

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
lb <- round(length(debateCorpus)*.075, 0)
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

###########
moddf <- data.frame(as.matrix(debateCorpus.clean.tfidf))
moddf$y <- as.factor(debate$Speaker)

mod1 <- glm(y ~ . , data = moddf, family = binomial(link='logit'))

moddf$preds1 <- predict(mod1, type = 'response')
#View(moddf[, c('y', 'preds1')])

# create an index
moddf$index <- as.numeric(row.names(moddf))
moddf <- moddf[order(moddf$preds1), ]
moddf$probOrder <- seq(1,nrow(moddf))

moddf$rowSums <- rowSums(subset(moddf, select = -c(y,preds1, index, probOrder)))

View(moddf[moddf$rowSums==0, c('y', 'preds1')])

##### PLOT
library(ggplot2)
# un-ordered
#ggplot(moddf[moddf$rowSums> 0, ], aes(x=index, y=preds1, colour = y)) + geom_point()

# ordered by probability
ggplot(moddf[moddf$rowSums> 0, ], aes(x=probOrder, y=preds1, colour = y)) + geom_point()






df = data.frame(c1=c('a','b'),c2=c(1,2)) 
colnames(df)[which(df == "b", arr.ind = TRUE)[2]] 




