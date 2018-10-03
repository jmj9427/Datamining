# Data Mining (2017.05~2017.07)
# Project goal
# Group classification

#####################################
word2vec                            #
#####################################
##### Required packages

# Install & load word2vec package
if(!require(devtools)) install.packages("devtools"); library(devtools)
if(!require(wordVectors)) install_github("bmschmidt/wordVectors"); library(wordVectors)

# Install & load data.table package
if(!require(data.table)) install.packages("data.table"); library(data.table)
if(!require(randomForest)) install.packages("randomForest"); library(randomForest)
if(!require(caret)) install.packages("caret"); library(caret)

# list objects in word2vec package
ls("package:wordVectors")

### 안되면 이걸로 
library(devtools)
devtools::install_github("bmschmidt/wordVectors")

### Data Load
cs.dt <- fread("train_profiles.csv")
cs.dt$GENDER<-substr(cs.dt$GROUP, 1, 1)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID) ; setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
head(md.dt)
md.dt$GENDER<-substr(md.dt$GROUP, 1, 1)

###### Make sites sentences _ GENDER
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, GENDER][1]
  itemfreq <- table(md.dt[CUS_ID==x,  ACT_NM])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "items.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(12345)
model = train_word2vec("items.txt","vec.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,GENDER])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,GENDER]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,GENDER]), unique(md.dt[CUS_ID==1, ACT_NM]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

##train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, ACT_NM]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, ACT_NM]), average=T]], model[[c("M","F"), average=F]])

##test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, ACT_NM]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, ACT_NM]), average=T]], model[[c("M","F"), average=F]])

#### train _ word2vec
result=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, ACT_NM]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result=rbind(result,c)
}
head(result)
save(result, file="result.rda")

###### test _ word2vec
result1=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, ACT_NM]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result1=rbind(result1,c)
}
head(result1)
save(result1, file="result1.rda")


cs.dt <- fread("train_profiles.csv")
cs.dt$age<-substr(cs.dt$GROUP, 2, 3)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]

head(cs.dt)
head(tr.dt)
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)

###### Make sites sentences _ AGE
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, age][1]
  itemfreq <- table(md.dt[CUS_ID==x,  ACT_NM])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "age.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("age.txt","vec2.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec2.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,age])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,age]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,age]), unique(md.dt[CUS_ID==1, ACT_NM]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, ACT_NM]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, ACT_NM]), average=T]], model[[c("20","30","40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==1, ACT_NM]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2, ACT_NM]), average=T]], model[[c("20","30","40"), average=F]])

result3=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, ACT_NM]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result3=rbind(result3,c)
}
head(result3)
save(result3, file="result3.rda")


result4=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, ACT_NM]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result4=rbind(result4,c)
}
head(result4)
save(result4, file="result4.rda")



cs.dt <- fread("train_profiles.csv")
cs.dt$GROUP<-substr(cs.dt$GROUP, 1, 3)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]

head(cs.dt)
head(tr.dt)
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences _ AGE + GENDER
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, GROUP][1]
  itemfreq <- table(md.dt[CUS_ID==x,  ACT_NM])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "group.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("group.txt","vec3.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec3.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,GROUP])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,GROUP]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,GROUP]), unique(md.dt[CUS_ID==1, ACT_NM]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, ACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, ACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, ACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, ACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

result5=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, ACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result5=rbind(result5,c)
}
head(result5)
save(result5, file="result5.rda")


result6=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, ACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result6=rbind(result6,c)
}
head(result6)
save(result6, file="result6.rda")


cs.dt <- fread("train_profiles.csv")
cs.dt$GROUP<-substr(cs.dt$GROUP, 1, 1)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]

setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, GROUP][1]
  itemfreq <- table(md.dt[CUS_ID==x,  SITE_NM])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "0518_sex.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("0518_sex.txt","vec0518_sex.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec0518_sex.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,GROUP])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,GROUP]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,GROUP]), unique(md.dt[CUS_ID==1, SITE_NM]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, SITE_NM]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, SITE_NM]), average=T]], model[[c("M","F"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, SITE_NM]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, SITE_NM]), average=T]], model[[c("M","F"), average=F]])

result9=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, SITE_NM]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result9=rbind(result9,c)
}
head(result9)
save(result9, file="result9.rda")


result10=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, SITE_NM]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result10=rbind(result10,c)
}
head(result10)
save(result10, file="result10.rda")

cs.dt <- fread("train_profiles.csv")
cs.dt$age<-substr(cs.dt$GROUP, 2, 3)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]

head(cs.dt)
head(tr.dt)
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, age][1]
  itemfreq <- table(md.dt[CUS_ID==x,  SITE_NM])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "0518_age.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("0518_age.txt","vec0518_age.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec0518_age.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,age])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,age]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,age]), unique(md.dt[CUS_ID==1, SITE_NM]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, SITE_NM]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, SITE_NM]), average=T]], model[[c("20","30","40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, SITE_NM]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, SITE_NM]), average=T]], model[[c("20","30","40"), average=F]])

result11=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, SITE_NM]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result11=rbind(result11,c)
}
head(result11)
save(result11, file="result11.rda")


result12=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, SITE_NM]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result12=rbind(result12,c)
}
head(result12)
save(result12, file="result12.rda")


cs.dt <- fread("train_profiles.csv")
cs.dt$GROUP<-substr(cs.dt$GROUP, 1, 3)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]


setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, GROUP][1]
  itemfreq <- table(md.dt[CUS_ID==x,  SITE_NM])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "0518.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("0518.txt","vec0518.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec0518.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,GROUP])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,GROUP]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,GROUP]), unique(md.dt[CUS_ID==1, SITE_NM]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, SITE_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, SITE_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, SITE_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, SITE_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

result7=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, SITE_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result7=rbind(result7,c)
}
head(result7)
save(result7, file="result7.rda")


result8=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, SITE_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result8=rbind(result8,c)
}
head(result8)
save(result8, file="result8.rda")


cs.dt <- fread("train_profiles.csv")
cs.dt$GROUP<-substr(cs.dt$GROUP, 1, 1)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]


setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, GROUP][1]
  itemfreq <- table(md.dt[CUS_ID==x,  SITE])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "0518_sex.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("0518_sex1.txt","vec0518_sex1.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec0518_sex1.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,GROUP])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,GROUP]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,GROUP]), unique(md.dt[CUS_ID==1, SITE]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, SITE]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, SITE]), average=T]], model[[c("M","F"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, SITE]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, SITE]), average=T]], model[[c("M","F"), average=F]])

result20=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, SITE]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result20=rbind(result20,c)
}
head(result20)
save(result20, file="result20.rda")


result21=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, SITE]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result21=rbind(result21,c)
}
head(result21)
save(result21, file="result21.rda")


cs.dt <- fread("train_profiles.csv")
cs.dt$age<-substr(cs.dt$GROUP, 2, 3)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]

head(cs.dt)
head(tr.dt)
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, age][1]
  itemfreq <- table(md.dt[CUS_ID==x,  SITE])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "0518_age.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("0518_age1.txt","vec0518_age1.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec0518_age1.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,age])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,age]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,age]), unique(md.dt[CUS_ID==1, SITE]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, SITE]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, SITE]), average=T]], model[[c("20","30","40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, SITE]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, SITE]), average=T]], model[[c("20","30","40"), average=F]])

result22=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, SITE]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result22=rbind(result22,c)
}
head(result22)
save(result22, file="result22.rda")


result23=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, SITE]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result23=rbind(result23,c)
}
head(result23)
save(result23, file="result23.rda")


cs.dt <- fread("train_profiles.csv")
cs.dt$GROUP<-substr(cs.dt$GROUP, 1, 3)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]


setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, GROUP][1]
  itemfreq <- table(md.dt[CUS_ID==x,  SITE])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "0518.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("0518.txt","vec0518.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec0518.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,GROUP])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,GROUP]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,GROUP]), unique(md.dt[CUS_ID==1, SITE]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, SITE]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, SITE]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, SITE]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, SITE]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

result24=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, SITE]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result24=rbind(result24,c)
}
head(result24)
save(result24, file="result24.rda")


result25=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, SITE]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result25=rbind(result25,c)
}
head(result25)
save(result25, file="result25.rda")

cs.dt <- fread("train_profiles.csv")
cs.dt$gender<-substr(cs.dt$GROUP, 1, 1)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]


setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, gender][1]
  itemfreq <- table(md.dt[CUS_ID==x,  MACT_NM])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "0518_sex.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("0518_sex.txt","vec0518_sex.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec0518_sex.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,gender])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,gender]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[gender]), unique(md.dt[CUS_ID==1, MACT_NM]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, MACT_NM]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, MACT_NM]), average=T]], model[[c("M","F"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, MACT_NM]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, MACT_NM]), average=T]], model[[c("M","F"), average=F]])

result9=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, MACT_NM]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result9=rbind(result9,c)
}
head(result9)
save(result9, file="result_mact_gender_train.rda")


result10=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, MACT_NM]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result10=rbind(result10,c)
}
head(result10)
save(result10, file="result_mact_gender_test.rda")



cs.dt <- fread("train_profiles.csv")
cs.dt$age<-substr(cs.dt$GROUP, 2, 3)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]


setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, age][1]
  itemfreq <- table(md.dt[CUS_ID==x,  MACT_NM])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "0518_age.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("0518_age.txt","vec0518_age.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec0518_age.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,age])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,age]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,age]), unique(md.dt[CUS_ID==1, MACT_NM]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, MACT_NM]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, MACT_NM]), average=T]], model[[c("20","30","40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, MACT_NM]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, MACT_NM]), average=T]], model[[c("20","30","40"), average=F]])

result9=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, MACT_NM]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result9=rbind(result9,c)
}
head(result9)
save(result9, file="result_mact_age_train.rda")


result10=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, MACT_NM]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result10=rbind(result10,c)
}
head(result10)
save(result10, file="result_mact_age_test.rda")

cs.dt <- fread("train_profiles.csv")
cs.dt$GROUP<-substr(cs.dt$GROUP, 1, 3)
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
ts.dt <- fread("test_clickstreams.tab"); ts.dt[,CUS_ID:= as.numeric(CUS_ID)]


setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID); setkey(ts.dt, CUS_ID)
md.dt <- merge(cs.dt, tr.dt)
###### Make sites sentences
f <- function(x, t) {
  grp <- md.dt[CUS_ID==x, GROUP][1]
  itemfreq <- table(md.dt[CUS_ID==x,  MACT_NM])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "0518_group.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("0518_group.txt","vec0518_group.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec0518_group.bin") # reload the model. 

##### Explore the model
for (v in unique(md.dt[,GROUP])) print(closest_to(model, v, n=10))
model[[unique(md.dt[,GROUP]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(md.dt[,GROUP]), unique(md.dt[CUS_ID==1, MACT_NM]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(md.dt[CUS_ID==1, MACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(md.dt[CUS_ID==2, MACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2501, MACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(ts.dt[CUS_ID==2502, MACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

result9=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(md.dt[CUS_ID==i, MACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result9=rbind(result9,c)
}
head(result9)
save(result9, file="result_mact_group_train.rda")


result10=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(ts.dt[CUS_ID==i, MACT_NM]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result10=rbind(result10,c)
}
head(result10)
save(result10, file="result_mact_group_test.rda")

#######################################################################################################
################## keyword gender, age , group 코사인유사도############
### gender word2vec
library(data.table)
cs.dt <- fread("train_profiles.csv")
cs.dt2 <- fread("sample_submission.csv")
ts<-read.delim("train_searchkeywords.tab",stringsAsFactors = F);ts[,CUS_ID:= as.numeric(CUS_ID)]
ts2<-read.delim("test_searchkeywords.tab",stringsAsFactors = F);ts2[,CUS_ID:= as.numeric(CUS_ID)]
setkey(cs.dt, CUS_ID); setkey(ts, CUS_ID) 
setkey(cs.dt2, CUS_ID); setkey(ts2, CUS_ID)
ts$CUS_ID=as.numeric(ts$CUS_ID)
ts2$CUS_ID=as.numeric(ts2$CUS_ID)
mg<-merge(cs.dt,ts,by="CUS_ID",all.x=T)
mg2<-merge(cs.dt2,ts2,by="CUS_ID",all.x=T)
mg2=mg2[,-2:-7]
mg$QRY_STR<-ifelse(regexpr("query",mg$QRY_STR)>0 ,
                   substr(mg$QRY_STR,regexpr("query",mg$QRY_STR)+6,500),
                   mg$QRY_STR)
mg$QRY_STR<-ifelse(regexpr("acq",mg$QRY_STR)>0 ,
                   substr(mg$QRY_STR,1,regexpr("&",mg$QRY_STR)-1),
                   mg$QRY_STR)
mg[is.na(mg)]=0

mg2$QRY_STR<-ifelse(regexpr("query",mg2$QRY_STR)>0 ,
                    substr(mg2$QRY_STR,regexpr("query",mg2$QRY_STR)+6,500),
                    mg2$QRY_STR)
mg2$QRY_STR<-ifelse(regexpr("acq",mg2$QRY_STR)>0 ,
                    substr(mg2$QRY_STR,1,regexpr("&",mg2$QRY_STR)-1),
                    mg2$QRY_STR)
mg2[is.na(mg2)]=0

mg$QRY_STR = gsub('pre \\d+', '', mg$QRY_STR)
mg$QRY_STR = gsub('pre', '', mg$QRY_STR)
mg$QRY_STR = gsub('qdt \\d+', '', mg$QRY_STR)
mg$QRY_STR = gsub('qdt', '', mg$QRY_STR)
mg$QRY_STR = gsub('query \\d+', '', mg$QRY_STR)
mg$QRY_STR = gsub('query', '', mg$QRY_STR)
mg$QRY_STR = gsub('sm \\d+', '', mg$QRY_STR)
mg$QRY_STR = gsub('sug \\d+', '', mg$QRY_STR)
mg$QRY_STR = gsub('sug', '', mg$QRY_STR)
mg$QRY_STR = gsub('top \\d+', '', mg$QRY_STR)
mg$QRY_STR = gsub('top', '', mg$QRY_STR)
mg$QRY_STR = gsub('utf\\d+', '', mg$QRY_STR)
mg$QRY_STR = gsub('utf', '', mg$QRY_STR)

mg = mg[!(mg$QRY_STR == "(?<=[\\s])\\s*|^\\s+|\\s+$"),]


mg2$QRY_STR = gsub('pre \\d+', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('pre', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('qdt \\d+', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('qdt', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('query \\d+', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('query', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('sm \\d+', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('sug \\d+', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('sug', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('top \\d+', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('top', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('utf\\d+', '', mg2$QRY_STR)
mg2$QRY_STR = gsub('utf', '', mg2$QRY_STR)

mg2 = mg2[!(mg2$QRY_STR == "(?<=[\\s])\\s*|^\\s+|\\s+$"),]

write.csv(mg,"mg.csv")
write.csv(mg2,"mg2.csv")


mg=read.csv("mg.csv",sep=",")
mg=mg[,-1]
mg2=read.csv("mg2.csv",sep=",")
mg2=mg2[,-1]
mg=data.table(mg)
mg2=data.table(mg2)

mg$GENDER<-substr(mg$GROUP, 1, 1)
###### Make sites sentences
f <- function(x, t) {
  grp <- mg[CUS_ID==x, GENDER][1]
  itemfreq <- table(mg[CUS_ID==x,  QRY_STR])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "items.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(12345)
model = train_word2vec("items.txt","vec.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec.bin") # reload the model. 
save(model,file="MODEL.word.keyword(300,gender).rda")


##### Explore the model
for (v in unique(mg[,GENDER])) print(closest_to(model, v, n=10))
model[[unique(mg[,GENDER]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(mg[,GENDER]), unique(mg[CUS_ID==1, QRY_STR]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

##train cosine 유사도
cosineSimilarity(model[[unique(mg[CUS_ID==1, QRY_STR]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(mg[CUS_ID==2, QRY_STR]), average=T]], model[[c("M","F"), average=F]])

##test cosine 유사도
cosineSimilarity(model[[unique(mg2[CUS_ID==2501, QRY_STR]), average=T]], model[[c("M","F"), average=F]])
cosineSimilarity(model[[unique(mg2[CUS_ID==2502, QRY_STR]), average=T]], model[[c("M","F"), average=F]])

#### train _ word2vec
result=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(mg[CUS_ID==i, QRY_STR]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result=rbind(result,c)
}
head(result)
result[is.na(result)]=0
result50=result
save(result50, file="result50.rda")

###### test _ word2vec
result1=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(mg2[CUS_ID==i, QRY_STR]), average=T]], model[[c("M","F"), average=F]])
  c <- data.frame(a,b)
  result1=rbind(result1,c)
}
head(result1)
result1[is.na(result1)]=0
result51=result1
save(result51, file="result51.rda")

##### Predict & Evaluate
# calculate the cosine similarity between items and target classes
g <- function(x, dt, min) {
  itemfreq <- table(dt[CUS_ID==x, QRY_STR])
  fitems <- itemfreq[itemfreq >= min]
  sim <- cosineSimilarity(model[[names(fitems), average=T]], model[[c("M","F"), average=F]])
  return(names(which.max(sim[1,])))
}
# accuracy for train data
ctab <- table(sapply(mg$CUS_ID, g, mg, 1), mg$GENDER); ctab
sum(diag(ctab)) / nrow(mg) 
nrow(mg[GENDER=="M",]) / nrow(mg)
# Training accuracy = 74.28% (NIR: 63.44%)




### age word2vec

mg$age<-substr(mg$GROUP, 2, 3)

###### Make sites sentences
f <- function(x, t) {
  grp <- mg[CUS_ID==x, age][1]
  itemfreq <- table(mg[CUS_ID==x,  QRY_STR])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "age.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("age.txt","vec2.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec2.bin") # reload the model. 
save(model,file="MODEL.word.keyword(300,age).rda")

##### Explore the model
for (v in unique(mg[,age])) print(closest_to(model, v, n=10))
model[[unique(mg[,age]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(mg[,age]), unique(mg[CUS_ID==1, QRY_STR]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(mg[CUS_ID==1, QRY_STR]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(mg[CUS_ID==2, QRY_STR]), average=T]], model[[c("20","30","40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(mg2[CUS_ID==2501, QRY_STR]), average=T]], model[[c("20","30","40"), average=F]])
cosineSimilarity(model[[unique(mg2[CUS_ID==2502, QRY_STR]), average=T]], model[[c("20","30","40"), average=F]])

result3=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(mg[CUS_ID==i, QRY_STR]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result3=rbind(result3,c)
}
head(result3)
result52=result3
result52[is.na(result52)]=0
save(result52, file="result52.rda")


result4=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(mg2[CUS_ID==i, QRY_STR]), average=T]], model[[c("20","30","40"), average=F]])
  c <- data.frame(a,b)
  result4=rbind(result4,c)
}
head(result4)
result53=result4
result53[is.na(result53)]=0
save(result53, file="result53.rda")



### GROUP word2vec


mg$GROUP<-substr(mg$GROUP, 1, 3)



###### Make sites sentences
f <- function(x, t) {
  grp <- mg[CUS_ID==x, GROUP][1]
  itemfreq <- table(mg[CUS_ID==x,  QRY_STR])
  fitems <- itemfreq[itemfreq >= t]
  act <- names(fitems)
  #  
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  #
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 2))
write.table(items, "group.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Train site2vec model
set.seed(1234)
model = train_word2vec("group.txt","vec3.bin",vectors=300,threads=1,window=5,cbow=1,iter=5,negative_samples=10, force = T)
model <- read.binary.vectors("vec3.bin") # reload the model. 
save(model,file="MODEL.word.keyword(300,group).rda")

##### Explore the model
for (v in unique(mg[,GROUP])) print(closest_to(model, v, n=10))
model[[unique(mg[,GROUP]), average=F]] %>% plot(method="pca")
items.1 <- c(unique(mg[,GROUP]), unique(mg[CUS_ID==1, QRY_STR]))
model[[items.1[1:10], average=F]] %>% plot(method="pca")

## train cosine 유사도
cosineSimilarity(model[[unique(mg[CUS_ID==1, QRY_STR]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(mg[CUS_ID==2, QRY_STR]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

## test cosine 유사도
cosineSimilarity(model[[unique(mg2[CUS_ID==2501, QRY_STR]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
cosineSimilarity(model[[unique(mg2[CUS_ID==2502, QRY_STR]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])

result5=NULL
for (i in 1:2500){
  a <- i
  b <- cosineSimilarity(model[[unique(mg[CUS_ID==i, QRY_STR]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result5=rbind(result5,c)
}
head(result5)
result54=result5
result54[is.na(result54)]=0
save(result54, file="result54.rda")


result6=NULL
for (i in 2501:5000){
  a <- i
  b <- cosineSimilarity(model[[unique(mg2[CUS_ID==i, QRY_STR]), average=T]], model[[c("F20","F30","F40","M20","M30","M40"), average=F]])
  c <- data.frame(a,b)
  result6=rbind(result6,c)
}
head(result6)
result55=result6
result55[is.na(result55)]=0
save(result55, file="result55.rda")
##########################################################################################################
##########
LDA   ####
##########
###################################### lda ACT_NM,SITE_NM
# train data 
cs.dt <- fread("train_profiles.csv")
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID) 
md.dt <- merge(cs.dt, tr.dt)

# test data
tr.t.dt <- fread("test_clickstreams.tab"); tr.t.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(tr.t.dt, CUS_ID)

###### Make Corpus (sites sentences)

f <- function(x, dt) {
  itemfreq <- table(dt[CUS_ID==x, ACT_NM])
  fitems <- itemfreq[itemfreq >= 1]
  act <- names(fitems)
  return(paste(act, collapse = " "))
}
md.dt$ACT_NM <- gsub(" ", "_", md.dt$ACT_NM); 
tr.t.dt$ACT_NM <- gsub(" ", "_", tr.t.dt$ACT_NM)
items <- unlist(sapply(cs.dt$CUS_ID, f, md.dt))
items <- c(items, unlist(sapply(unique(tr.t.dt$CUS_ID), f, tr.t.dt)))

##### create the document term matrix (DTM)
# DTM is a mathematical matrix that describes the frequency of terms that occur in a collection of documents.
# In a document-term matrix, rows correspond to documents in the collection and columns correspond to terms. 

tic <- proc.time()
items.dtm <- DocumentTermMatrix(Corpus(VectorSource(items)))
print(proc.time() - tic)

##### Run LDA model

tic <- proc.time()
lda.model <- LDA(items.dtm, k=20, method="Gibbs", control=list(burnin=1000, iter=1000, keep=50))
save(lda.model,file="lda.model.rda")

print(proc.time() - tic)
# Saving and loading lda model: 
# saveRDS(lda.model, "lda_model.rds") 
# lda.model <- readRDS("lda_model.rds")


##### Calculate the per document probabilities of the topics

items.theta <- as.data.frame(posterior(lda.model)$topics)
#head(items.theta[1:5])
train <- cbind(data.frame(CUS_ID=cs.dt$CUS_ID), items.theta[1:2500,])

# for test data
test.CUS_ID <- unique(tr.t.dt$CUS_ID)
test <- cbind(data.frame(CUS_ID=test.CUS_ID), items.theta[2501:5000,])


a<-NULL
for (i in 0:19) { 
  a[i+1] <- paste("ACT", i+1, sep="")
}

colnames(train) <- c("CUS_ID",a)
colnames(test) <- c("CUS_ID",a)
save(train,file="lda.train.rda")
save(test,file="lda.test.rda")



######## SITE_NM ###########

f <- function(x, dt) {
  itemfreq <- table(dt[CUS_ID==x, SITE_NM])
  fitems <- itemfreq[itemfreq >= 1]
  SITE <- names(fitems)
  return(paste(SITE, collapse = " "))
}
md.dt$SITE_NM <- gsub(" ", "_", md.dt$SITE_NM); 
tr.t.dt$SITE_NM <- gsub(" ", "_", tr.t.dt$SITE_NM)
items <- unlist(sapply(cs.dt$CUS_ID, f, md.dt))
items <- c(items, unlist(sapply(unique(tr.t.dt$CUS_ID), f, tr.t.dt)))

items.dtm <- DocumentTermMatrix(Corpus(VectorSource(items)))

lda.model <- LDA(items.dtm, k=20, method="Gibbs", control=list(burnin=1000, iter=1000, keep=50))

items.theta <- as.data.frame(posterior(lda.model)$topics)
#head(items.theta[1:5])
train2 <- cbind(data.frame(CUS_ID=cs.dt$CUS_ID), items.theta[1:2500,], data.frame(GROUP=make.names(cs.dt$GROUP)))

# for test data
test.CUS_ID <- unique(tr.t.dt$CUS_ID)
test2 <- cbind(data.frame(CUS_ID=test.CUS_ID), items.theta[2501:5000,])

a<-NULL
for (i in 0:19) { 
  a[i+1] <- paste("SITE", i+1, sep="")
}

colnames(train2) <- c("CUS_ID",a)
colnames(test2) <- c("CUS_ID",a)
save(train2,file="lda.train2.rda")
save(test2,file="lda.test2.rda")


##################### rf 모델
load("lda.train.rda")
load("lda.test.rda")
load("lda.train2.rda")
load("lda.test2.rda")

train=train[,-1]
test=test[,-1]
train2=train2[,c(-1,-22)]
test2=test2[,-1]

load("result.rda")
load("result1.rda")
load("result3.rda")
load("result4.rda")
load("result5.rda")
load("result6.rda")
load("result7.rda")
load("result8.rda")
load("result9.rda")
load("result10.rda")
load("result11.rda")
load("result12.rda")
load("result20.rda")
load("result21.rda")
load("result22.rda")
load("result23.rda")
load("result24.rda")
load("result25.rda")
load("result50.rda")
load("result51.rda")
load("result52.rda")
load("result53.rda")
load("result54.rda")
load("result55.rda")
load("result_mact_group_test.rda")
load("result_mact_group_train.rda")
load("result_mact_age_test.rda")
load("result_mact_age_train.rda")
load("result_mact_gender_train.rda")
load("result_mact_gender_test.rda")

train.r=data.frame(result,result3,result5,result7,result9,result11,result20,result22,result24)
test.r=data.frame(result1,result4,result6,result8,result10,result12,result21,result23,result25)
test.r=test.r[,c(-1,-4,-8,-15,-22,-25,-29,-32,-36)]
train.r=train.r[,c(-1,-4,-8,-15,-22,-25,-29,-32,-36)]
colnames(train.r)=c("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","X24","X25","X26","X27","X28","X29","X30","X31","X32","X33")
colnames(test.r)=c("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","X24","X25","X26","X27","X28","X29","X30","X31","X32","X33")


train.r2=data.frame(result50,result52,result54)
test.r2=data.frame(result51,result53,result55)
train.r2=train.r2[,c(-1,-4,-8)]
test.r2=test.r2[,c(-1,-4,-8)]
colnames(train.r2)=c("X34","X35","X36","X37","X38","X39","X40","X41","X42","X43","X44")
colnames(test.r2)=c("X34","X35","X36","X37","X38","X39","X40","X41","X42","X43","X44")

train.r3=data.frame(result_mact_gender,result_mact_age,result_mact_group)
test.r3=data.frame(result_mact_gender_test,result_mact_age_test,result_mact_group_test)
train.r3=train.r3[,c(-1,-4,-8)]
test.r3=test.r3[,c(-1,-4,-8)]
colnames(train.r3)=c("X45","X46","X47","X48","X49","X50","X51","X52","X53","X54","X55")
colnames(test.r3)=c("X45","X46","X47","X48","X49","X50","X51","X52","X53","X54","X55")


train.r4=cbind(train.r,train.r2,train.r3)
test.r4=cbind(test.r,test.r2,test.r3)

cs.dt <- fread("train_profiles.csv")
cs.dt=cs.dt[order(cs.dt$CUS_ID),]
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID) 
md.dt <- merge(cs.dt, tr.dt)

# test data


tr.t.dt=fread("sample_submission.csv")
setkey(tr.t.dt, CUS_ID)

train=cbind(train,train2,train.r4)
test=cbind(test,test2,test.r4)

train=cbind(CUS_ID=cs.dt$CUS_ID,train,GROUP=cs.dt$GROUP)
test=cbind(CUS_ID=unique(tr.t.dt$CUS_ID),test)


test.public <- read.csv("test_public.csv", stringsAsFactors = F)
tfi<-merge(test.public,test,by="CUS_ID")
tfi$GENDER<-ifelse(tfi$F20. | tfi$F30 | tfi$F40. ==1,"F","M")
tfi$AGE<-ifelse(tfi$F20. | tfi$M20. ==1,"20-",ifelse(tfi$F30 | tfi$M30 ==1,"30","40+"))
tfi$GROUP<-paste(tfi$GENDER,tfi$AGE,sep="")
tfi=tfi[,c(-103,-104)]

tfi2<-tfi[,-c(2:7)]
train<-rbind(train,tfi2)
str(train)
str(test)
train$GROUP=substr(train$GROUP,1,3)


control <- trainControl(method="cv", number=5, repeats=1, classProbs=TRUE, summaryFunction=mnLogLoss)

# Used models
methods <- c("rf") # add methods such as xgbTree, rf, svmRadious, etc.

models <- list()
for (i in 1:length(methods)) {
  set.seed(123)
  model <- train(GROUP ~ .,
                 data = subset(train, select=-CUS_ID),
                 method = methods[i],
                 preProcess = NULL,
                 metric = "logLoss",
                 trControl = control)
  models[[i]] <- model
}
names(models) <- methods
# Saving and loading models: 
# saveRDS(models, "models.rds") 
# models <- readRDS("models.rds")

# Model comparison
results <- resamples(models)
summary(results)
xyplot(results)
modelCor(results)
splom(results)

# prediction & submission
for (i in 1:length(methods)) {
  pred <- predict(models[i], test, type="prob")
  fname <- paste("submission_", methods[i], ".csv", sep="")
  write.csv(cbind(CUS_ID=test$CUS_ID,pred[[1]]), fname, row.names = F)
}
####################################################################################


##### Required packages

library("xgboost", lib.loc="D:/R-3.3.3/library")
library("wordVectors", lib.loc="D:/R-3.3.3/library")
library("tree", lib.loc="D:/R-3.3.3/library")
library("tidyr", lib.loc="D:/R-3.3.3/library")
library("reshape", lib.loc="D:/R-3.3.3/library")
library("reshape2", lib.loc="D:/R-3.3.3/library")
library("randomForest", lib.loc="D:/R-3.3.3/library")
library("nnet", lib.loc="D:/R-3.3.3/library")
library("caret", lib.loc="D:/R-3.3.3/library")
library("caretEnsemble", lib.loc="D:/R-3.3.3/library")
library("data.table", lib.loc="D:/R-3.3.3/library")
library("e1071", lib.loc="D:/R-3.3.3/library")
library("gbm", lib.loc="D:/R-3.3.3/library")
library("glmnet", lib.loc="D:/R-3.3.3/library")
library("kernlab", lib.loc="D:/R-3.3.3/library")


############SITE_NM(vector 200)################

# train data
cs.dt <- fread("train_profiles.csv")
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID) 
md.dt <- merge(cs.dt, tr.dt)

# test data
tr.t.dt <- fread("test_clickstreams.tab"); tr.t.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(tr.t.dt, CUS_ID)

###### Make Corpus (sites sentences)

f <- function(x, min) {
  # Select sites accessed min times and more  
  grp <- cs.dt[CUS_ID==x, GROUP]
  itemfreq <- table(md.dt[CUS_ID==x, SITE_NM])
  fitems <- itemfreq[itemfreq >= min]
  act <- names(fitems)
  # Replace blanks in ACT_NM with underscore
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  # Boost transactions 
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 1)) # best performed when min = 1
write.table(items, "items.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Build trans2vec model
set.seed(12345)
model = train_word2vec("items.txt","vec.bin",vectors=200,threads=1,window=5,cbow=1,negative_samples=10,iter=5,force = T)
# model <- read.binary.vectors("vec.bin") # reload the pre-trained word2vec model 
save(model,file="MODEL.word(SITE_NM,200).rda")



##### Prediction using trans2vec + classifaction methods

### Make features (mean vector)
# Get mean vector
g <- function(x, dt) {
  items <- dt[CUS_ID==x, SITE_NM]
  mvector <- model[[items, average=T]]
  return(mvector)
}
# for train data
fv <- t(sapply(cs.dt$CUS_ID, g, md.dt))
train <- cbind(data.frame(CUS_ID=cs.dt$CUS_ID), as.data.frame(fv), data.frame(GROUP=make.names(cs.dt$GROUP)))
# for test data
test.CUS_ID <- unique(tr.t.dt$CUS_ID)
fv <- t(sapply(test.CUS_ID, g, tr.t.dt))
test <- cbind(data.frame(CUS_ID=test.CUS_ID), as.data.frame(fv))


train.s2=train
test.s2=test
save(train.s2,file="train.s(200).rda")
save(test.s2,file="test.s(200).rda")

load("train.s(200).rda") #SITE_NM(vector 200) train
load("test.s(200).rda") #SITE_NM(vector 200) test

train.s2=train.s2[,c(-1,-202)]
test.s2=test.s2[,c(-1)]

train.s2=cbind(train.s2,train.r)
test.s2=cbind(test.s2,test.r)


train=train.s2
test=test.s2

train=cbind(CUS_ID=cs.dt$CUS_ID,train,GROUP=cs.dt$GROUP)
test=cbind(CUS_ID=unique(tr.t.dt$CUS_ID),test)
############test40%+train###########

test.public <- read.csv("test_public.csv", stringsAsFactors = F)
tfi<-merge(test.public,test,by="CUS_ID")
tfi$GENDER<-ifelse(tfi$F20. | tfi$F30 | tfi$F40. ==1,"F","M")
tfi$AGE<-ifelse(tfi$F20. | tfi$M20. ==1,"20-",ifelse(tfi$F30 | tfi$M30 ==1,"30","40+"))
tfi$GROUP<-paste(tfi$GENDER,tfi$AGE,sep="")
tfi=tfi[,c(-241,-242)]

tfi2<-tfi[,-c(2:7)]
train<-rbind(train,tfi2)
str(train)
str(test)
train$GROUP=substr(train$GROUP,1,3)

control <- trainControl(method="cv", number=5, repeats=1, classProbs=TRUE, summaryFunction=mnLogLoss)

# Used models
methods <- c("nnet") # add methods such as xgbTree, rf, svmRadious, etc.

models <- list()
for (i in 1:length(methods)) {
  set.seed(123)
  model <- train(GROUP ~ .,
                 data = subset(train, select=-CUS_ID),
                 method = methods[i],
                 preProcess = NULL,
                 metric = "logLoss",
                 trControl = control)
  models[[i]] <- model
}
names(models) <- methods
# Saving and loading models: 
# saveRDS(models, "models.rds") 
# models <- readRDS("models.rds")

# Model comparison
results <- resamples(models)
summary(results)
xyplot(results)
modelCor(results)
splom(results)

# prediction & submission
for (i in 1:length(methods)) {
  pred <- predict(models[i], test, type="prob")
  fname <- paste("submission_", methods[i], ".csv", sep="")
  write.csv(cbind(CUS_ID=test$CUS_ID,pred[[1]]), fname, row.names = F)
}
##############################################

############SICT_NM(vector 300)################

# train data
cs.dt <- fread("train_profiles.csv")
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID) 
md.dt <- merge(cs.dt, tr.dt)

# test data
tr.t.dt <- fread("test_clickstreams.tab"); tr.t.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(tr.t.dt, CUS_ID)

###### Make Corpus (sites sentences)

f <- function(x, min) {
  # Select sites accessed min times and more  
  grp <- cs.dt[CUS_ID==x, GROUP]
  itemfreq <- table(md.dt[CUS_ID==x, SITE_NM])
  fitems <- itemfreq[itemfreq >= min]
  act <- names(fitems)
  # Replace blanks in ACT_NM with underscore
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  # Boost transactions 
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 1)) # best performed when min = 1
write.table(items, "items.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Build trans2vec model
set.seed(12345)
model = train_word2vec("items.txt","vec.bin",vectors=300,threads=1,window=5,cbow=1,negative_samples=10,iter=5,force = T)
# model <- read.binary.vectors("vec.bin") # reload the pre-trained word2vec model 
save(model,file="MODEL.word(SICT_NM,300).rda")



##### Prediction using trans2vec + classifaction methods

### Make features (mean vector)
# Get mean vector
g <- function(x, dt) {
  items <- dt[CUS_ID==x, SITE_NM]
  mvector <- model[[items, average=T]]
  return(mvector)
}
# for train data
fv <- t(sapply(cs.dt$CUS_ID, g, md.dt))
train <- cbind(data.frame(CUS_ID=cs.dt$CUS_ID), as.data.frame(fv), data.frame(GROUP=make.names(cs.dt$GROUP)))
# for test data
test.CUS_ID <- unique(tr.t.dt$CUS_ID)
fv <- t(sapply(test.CUS_ID, g, tr.t.dt))
test <- cbind(data.frame(CUS_ID=test.CUS_ID), as.data.frame(fv))


train.s=train
test.s=test
save(train.s,file="train.s(300).rda")
save(test.s,file="test.s(300).rda")

load("train.s(300).rda") #SITE_NM(vector 200) train
load("test.s(300).rda") #SITE_NM(vector 200) test


train.s=train.s[,c(-1,-302)]
test.s=test.s[,c(-1)]

train=cbind(train.s,train.r)
test=cbind(test.s,test.r)

train=cbind(CUS_ID=cs.dt$CUS_ID,train,GROUP=cs.dt$GROUP)
test=cbind(CUS_ID=unique(tr.t.dt$CUS_ID),test)

############test40%+train###########

test.public <- read.csv("test_public.csv", stringsAsFactors = F)
tfi<-merge(test.public,test,by="CUS_ID")
tfi$GENDER<-ifelse(tfi$F20. | tfi$F30 | tfi$F40. ==1,"F","M")
tfi$AGE<-ifelse(tfi$F20. | tfi$M20. ==1,"20-",ifelse(tfi$F30 | tfi$M30 ==1,"30","40+"))
tfi$GROUP<-paste(tfi$GENDER,tfi$AGE,sep="")
tfi=tfi[,c(-341,-342)]

tfi2<-tfi[,-c(2:7)]
train<-rbind(train,tfi2)
str(train)
str(test)
train$GROUP=substr(train$GROUP,1,3)

control <- trainControl(method="cv", number=5, repeats=1, classProbs=TRUE, summaryFunction=mnLogLoss)

# Used models
methods <- c("xgbTree") # add methods such as xgbTree, rf, svmRadious, etc.

models <- list()
for (i in 1:length(methods)) {
  set.seed(123)
  model <- train(GROUP ~ .,
                 data = subset(train, select=-CUS_ID),
                 method = methods[i],
                 preProcess = NULL,
                 metric = "logLoss",
                 trControl = control)
  models[[i]] <- model
}
names(models) <- methods
# Saving and loading models: 
# saveRDS(models, "models.rds") 
# models <- readRDS("models.rds")

# Model comparison
results <- resamples(models)
summary(results)
xyplot(results)
modelCor(results)
splom(results)

# prediction & submission
for (i in 1:length(methods)) {
  pred <- predict(models[i], test, type="prob")
  fname <- paste("submission_", methods[i], ".csv", sep="")
  write.csv(cbind(CUS_ID=test$CUS_ID,pred[[1]]), fname, row.names = F)
}
#################################################################################################################################
install.packages("plyr")
install.packages("dplyr")
install.packages("data.table")
install.packages("caret")

library(plyr)
library(dplyr)
library(data.table)
library("caret")

pro = fread("train_profiles.csv")
pro_pub = fread("test_public.csv")
train_sk = fread("train_searchkeywords.tab") %>% mutate(QRY_STR = sub("&(.*)=", " ", QRY_STR))
test_sk = fread("test_searchkeywords.tab") %>% mutate(QRY_STR = sub("&(.*)=", " ", QRY_STR))

sk<-rbind(train_sk, test_sk)

#qry 정리
sk$QRY_STR <- gsub('pre \\d+', '', sk$QRY_STR)
sk$QRY_STR <- gsub('pre', '', sk$QRY_STR)

sk$QRY_STR <- gsub('qdt \\d+', '', sk$QRY_STR)
sk$QRY_STR <- gsub('qdt', '', sk$QRY_STR)

sk$QRY_STR <- gsub('query \\d+', '', sk$QRY_STR)
sk$QRY_STR <- gsub('query', '', sk$QRY_STR)

sk$QRY_STR <- gsub('sm \\d+', '', sk$QRY_STR)
sk$QRY_STR <- gsub('sm', '', sk$QRY_STR)

sk$QRY_STR <- gsub('sug \\d+', '', sk$QRY_STR)
sk$QRY_STR <- gsub('sug', '', sk$QRY_STR)

sk$QRY_STR <- gsub('top \\d+', '', sk$QRY_STR)
sk$QRY_STR <- gsub('top', '', sk$QRY_STR)

sk$QRY_STR <- gsub('utf\\d+', '', sk$QRY_STR)
sk$QRY_STR <- gsub('utf', '', sk$QRY_STR)

# 여러 공백이 존재하면 하나의 공백으로 변환
sk <- sk[!(sk$QRY_STR == "(?<=[\\s]\\s*|^\\s+|\\s+$"),]
save(sk, file = "sk.rda")

load("sk.rda")

##### Required packages

library(data.table)
library(reshape)
library(reshape2)
library(caret)
library(C50)
library(nnet)
library(devtools)
library(randomForest)
library(e1071)
library(devtools)
library(dplyr)

# Install & load word2vec package

if(!require(devtools)) install.packages("devtools"); library(devtools)
if(!require(wordVectors)) install_github("bmschmidt/wordVectors"); library(wordVectors)

# list objects in word2vec package

ls("package:wordVectors")

##########trans2vec QRY_STR

load("sk.rda") # train + test search data

# public -> 3503명 profile
public<-rbind(pro, pro_pub)

# cs.dt -> public
cs.dt<-public

# tr.dt -> sk
tr.dt <- sk

# md.dt -> merge(public, sk, by="CUS_ID)
md.dt<-merge(public, tr.dt, by="CUS_ID")

###### Make Corpus (sites sentences)

f <- function(x, min) {
  
  # Select sites accessed min times and more  
  
  grp <- cs.dt[CUS_ID==x, GROUP]
  
  itemfreq <- table(md.dt[CUS_ID==x, QRY_STR])
  
  fitems <- itemfreq[itemfreq >= min]
  
  qry <- names(fitems)
  
  # Replace blanks in STR_QRY with underscore
  
  sapply(qry, function(x) gsub(" ", "_", x))
  
  set.seed(1)
  
  # Boost transactions 
  
  as.vector((sapply(1:20, function(x) c(grp, sample(qry)))))
  
}

items <- unlist(sapply(cs.dt$CUS_ID, f, 1))
write.table(items, "item_3503_qry_trans2vec.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Build trans2vec model
set.seed(12345)
model = train_word2vec("item_3503_qry_trans2vec.txt","vec_3503_qry_trans2vec.bin",vectors=300,threads=1,window=5,cbow=1,negative_samples=10,iter=5,force = T)
model <- read.binary.vectors("vec_3503_qry_trans2vec.bin") # reload the pre-trained word2vec model 
head(model)

##### Prediction using trans2vec + classifaction methods

### Make features (mean vector)

# Get mean vector

g <- function(x, dt) {
  
  items <- dt[CUS_ID==x, QRY_STR]
  
  mvector <- model[[items, average=T]]
  
  return(mvector)
  
}

# for train data

fv <- t(sapply(cs.dt$CUS_ID, g, md.dt))

train <- cbind(data.frame(CUS_ID=cs.dt$CUS_ID), as.data.frame(fv), data.frame(GROUP=make.names(cs.dt$GROUP)))

head(train)

save(train, file="3503_group_qry_trans2vec_train.rda")

# make ts.dt(test data)
ts.dt = fread("test_searchkeywords.tab") %>% mutate(QRY_STR = sub("&(.*)=", " ", QRY_STR))

#qry 정리
ts.dt$QRY_STR <- gsub('pre \\d+', '', ts.dt$QRY_STR)
ts.dt$QRY_STR <- gsub('pre', '', ts.dt$QRY_STR)

ts.dt$QRY_STR <- gsub('qdt \\d+', '', ts.dt$QRY_STR)
ts.dt$QRY_STR <- gsub('qdt', '', ts.dt$QRY_STR)

ts.dt$QRY_STR <- gsub('query \\d+', '', ts.dt$QRY_STR)
ts.dt$QRY_STR <- gsub('query', '', ts.dt$QRY_STR)

ts.dt$QRY_STR <- gsub('sm \\d+', '', ts.dt$QRY_STR)
ts.dt$QRY_STR <- gsub('sm', '', ts.dt$QRY_STR)

ts.dt$QRY_STR <- gsub('sug \\d+', '', ts.dt$QRY_STR)
ts.dt$QRY_STR <- gsub('sug', '', ts.dt$QRY_STR)

ts.dt$QRY_STR <- gsub('top \\d+', '', ts.dt$QRY_STR)
ts.dt$QRY_STR <- gsub('top', '', ts.dt$QRY_STR)

ts.dt$QRY_STR <- gsub('utf\\d+', '', ts.dt$QRY_STR)
ts.dt$QRY_STR <- gsub('utf', '', ts.dt$QRY_STR)

# 여러 공백이 존재하면 하나의 공백으로 변환
ts.dt <- ts.dt[!(ts.dt$QRY_STR == "(?<=[\\s]\\s*|^\\s+|\\s+$"),]
save(ts.dt, file = "ts.dt.rda")
load("file=ts.dt.rda")

ts.dt<-data.table(ts.dt)

# for test data
test.CUS_ID <- unique(ts.dt$CUS_ID)

fv <- t(sapply(test.CUS_ID, g, ts.dt))

test <- cbind(data.frame(CUS_ID=test.CUS_ID), as.data.frame(fv))

head(test)

save(test, file="3503_group_qry_trans2vec_test.rda")

### Training & Prediction##################################################################################

# Control parameters for model training

control <- trainControl(method="cv", number=5, repeats=1, classProbs=TRUE, summaryFunction=mnLogLoss)

install.packages("gbm")

install.packages("xgboost")

install.packages("nnet")

install.packages("kernlab")

install.packages("randomForest")


library(gbm)

library(xgboost)

library(nnet)

library(kernlab)

library(randomForest)



# Used models

methods <- c("gbm", "nnet","svmRadial","xgbTree", "rf") # add methods such as xgbTree, rf, svmRadious, etc.



models <- list()

for (i in 1:length(methods)) {
  
  set.seed(123)
  
  model <- train(GROUP ~ .,
                 
                 data = subset(train, select=-CUS_ID),
                 
                 method = methods[i],
                 
                 preProcess = NULL,
                 
                 metric = "logLoss",
                 
                 trControl = control)
  
  models[[i]] <- model
  
}

names(models) <- methods





# Saving and loading models: 

saveRDS(models, "models3503_qry_group_trans2vec.rds") 

models <- readRDS("models0529.rds")



# Model comparison

results <- resamples(models)

summary(results)

xyplot(results)

modelCor(results)

splom(results)



# prediction & submission

for (i in 1:length(methods)) {
  
  pred <- predict(models[i], test, type="prob")
  
  fname <- paste("0530_qry_trans2vec_submission_", methods[i], ".csv", sep="") }

write.csv(cbind(CUS_ID=test.CUS_ID,pred[[1]]), fname, row.names = F) 

########################################################################################################################################
############ACT_NM(vector 300)################

# train data
cs.dt <- fread("train_profiles.csv")
tr.dt <- fread("train_clickstreams.tab"); tr.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(cs.dt, CUS_ID); setkey(tr.dt, CUS_ID) 
md.dt <- merge(cs.dt, tr.dt)

# test data
tr.t.dt <- fread("test_clickstreams.tab"); tr.t.dt[,CUS_ID:= as.numeric(CUS_ID)]
setkey(tr.t.dt, CUS_ID)

###### Make Corpus (sites sentences)

f <- function(x, min) {
  # Select sites accessed min times and more  
  grp <- cs.dt[CUS_ID==x, GROUP]
  itemfreq <- table(md.dt[CUS_ID==x, ACT_NM])
  fitems <- itemfreq[itemfreq >= min]
  act <- names(fitems)
  # Replace blanks in ACT_NM with underscore
  sapply(act, function(x) gsub(" ", "_", x))
  set.seed(1)
  # Boost transactions 
  as.vector((sapply(1:20, function(x) c(grp, sample(act)))))
}
items <- unlist(sapply(cs.dt$CUS_ID, f, 1)) # best performed when min = 1
write.table(items, "items.txt", eol = " ", quote = F, row.names = F, col.names = F)

##### Build trans2vec model
set.seed(12345)
model = train_word2vec("items.txt","vec.bin",vectors=300,threads=1,window=5,cbow=1,negative_samples=10,iter=5,force = T)
# model <- read.binary.vectors("vec.bin") # reload the pre-trained word2vec model 
save(model,file="MODEL.word(ACT,300).rda")


##### Prediction using trans2vec + classifaction methods

### Make features (mean vector)
# Get mean vector
g <- function(x, dt) {
  items <- dt[CUS_ID==x, ACT_NM]
  mvector <- model[[items, average=T]]
  return(mvector)
}
# for train data
fv <- t(sapply(cs.dt$CUS_ID, g, md.dt))
train <- cbind(data.frame(CUS_ID=cs.dt$CUS_ID), as.data.frame(fv), data.frame(GROUP=make.names(cs.dt$GROUP)))
# for test data
test.CUS_ID <- unique(tr.t.dt$CUS_ID)
fv <- t(sapply(test.CUS_ID, g, tr.t.dt))
test <- cbind(data.frame(CUS_ID=test.CUS_ID), as.data.frame(fv))


train.a=train
test.a=test
save(train.a,file="train.a(300).rda")
save(test.a,file="test.a(300).rda")


load("train.a(300).rda")
load("test.a(300).rda")
train=train.a
test=test.a
############test40%+train###########
cs.dt <- fread("train_profiles.csv")
cs.dt=cs.dt[order(cs.dt$CUS_ID),]
train$GROUP=cs.dt$GROUP
test.public <- read.csv("test_public.csv", stringsAsFactors = F)
tfi<-merge(test.public,test,by="CUS_ID")
tfi$GENDER<-ifelse(tfi$F20. | tfi$F30 | tfi$F40. ==1,"F","M")
tfi$AGE<-ifelse(tfi$F20. | tfi$M20. ==1,"20-",ifelse(tfi$F30 | tfi$M30 ==1,"30","40+"))
tfi$GROUP<-paste(tfi$GENDER,tfi$AGE,sep="")
tfi=tfi[,c(-308,-309)]

tfi2<-tfi[,-c(2:7)]
train<-rbind(train,tfi2)
str(train)
str(test)
train$GROUP=substr(train$GROUP,1,3)

### Training & Prediction
# Control parameters for model training
control <- trainControl(method="cv", number=5, repeats=1, classProbs=TRUE, summaryFunction=mnLogLoss)

# Used models
methods <- c("nnet") # add methods such as xgbTree, rf, svmRadious, etc.

models <- list()
for (i in 1:length(methods)) {
  set.seed(123)
  model <- train(GROUP ~ .,
                 data = subset(train, select=-CUS_ID),
                 method = methods[i],
                 preProcess = NULL,
                 metric = "logLoss",
                 trControl = control)
  models[[i]] <- model
}
names(models) <- methods
# Saving and loading models: 
# saveRDS(models, "models.rds") 
# models <- readRDS("models.rds")

# Model comparison
results <- resamples(models)
summary(results)
xyplot(results)
modelCor(results)
splom(results)

# prediction & submission
for (i in 1:length(methods)) {
  pred <- predict(models[i], test, type="prob")
  fname <- paste("submission_", methods[i], ".csv", sep="")
  write.csv(cbind(CUS_ID=test$CUS_ID,pred[[1]]), fname, row.names = F)
}
#########################################################################
############ACT_NM(vector 300) + 유사도################


load("train.a(300).rda")
load("test.a(300).rda")

train.a=train.a[,c(-1,-302)]
test.a=test.a[,c(-1)]

train=train.a
test=test.a
train=cbind(train,train.r)
test=cbind(test,test.r)
train=cbind(CUS_ID=cs.dt$CUS_ID,train,GROUP=cs.dt$GROUP)
test=cbind(CUS_ID=tr.t.dt$CUS_ID,test)
############test40%+train###########
cs.dt <- fread("train_profiles.csv")
cs.dt=cs.dt[order(cs.dt$CUS_ID),]
train$GROUP=cs.dt$GROUP
test.public <- read.csv("test_public.csv", stringsAsFactors = F)
tfi<-merge(test.public,test,by="CUS_ID")
tfi$GENDER<-ifelse(tfi$F20. | tfi$F30 | tfi$F40. ==1,"F","M")
tfi$AGE<-ifelse(tfi$F20. | tfi$M20. ==1,"20-",ifelse(tfi$F30 | tfi$M30 ==1,"30","40+"))
tfi$GROUP<-paste(tfi$GENDER,tfi$AGE,sep="")
tfi=tfi[,c(-341,-342)]

tfi2<-tfi[,-c(2:7)]
train<-rbind(train,tfi2)
str(train)
str(test)
train$GROUP=substr(train$GROUP,1,3)

### Training & Prediction
# Control parameters for model training
control <- trainControl(method="cv", number=5, repeats=1, classProbs=TRUE, summaryFunction=mnLogLoss)

# Used models
methods <- c("rf") # add methods such as xgbTree, rf, svmRadious, etc.

models <- list()
for (i in 1:length(methods)) {
  set.seed(123)
  model <- train(GROUP ~ .,
                 data = subset(train, select=-CUS_ID),
                 method = methods[i],
                 preProcess = NULL,
                 metric = "logLoss",
                 trControl = control)
  models[[i]] <- model
}
names(models) <- methods
# Saving and loading models: 
# saveRDS(models, "models.rds") 
# models <- readRDS("models.rds")

# Model comparison
results <- resamples(models)
summary(results)
xyplot(results)
modelCor(results)
splom(results)

# prediction & submission
for (i in 1:length(methods)) {
  pred <- predict(models[i], test, type="prob")
  fname <- paste("submission_", methods[i], ".csv", sep="")
  write.csv(cbind(CUS_ID=test$CUS_ID,pred[[1]]), fname, row.names = F)
}

##################################################################################
#Ensemble

#저희 조는 6개의 모델을 한꺼번에 앙상블했습니다.
#1. lda -> rf
#2. act -> nnet
#3. site -> xgbTree
#4. site -> nnet
#5. Qry -> xgbTree
#6. act -> rf
## PS. 5번 submission은 CUS_ID 가 2470개로 나와서 나머지는 엑셀로 처리.