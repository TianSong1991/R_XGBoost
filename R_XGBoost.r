rm(list=ls())
A<-read.table("E:\\test1.txt",header=T)
str(A)
head(A)
n<-nrow(A)
lie<-ncol(A)
data1<-A[,2:lie]
head(data1)

# 因子类型转换为数值型
# dataset2 = data1
# library(plyr)
# into_factor = function(x){
#   
#   if(class(x) == "factor"){
#     n = length(x)
#     data.fac = data.frame(x = x,y = 1:n)
#     output = model.matrix(y~x,data.fac)[,-1]
#     ## Convert factor into dummy variable matrix
#   }else{
#     output = x
#     ## if x is numeric, output is x
#   }
#   output
# }
# into_factor(data1$手次)[1:3,]
# dataset2 = colwise(into_factor)(dataset2)
# dataset2 = do.call(cbind,dataset2)
# dataset2 = as.data.frame(dataset2)
# head(dataset2)

#xgboost
library(xgboost)
library(ggplot2)
library(lattice)
library(minqa)
library(caret)
library(readr)
library(stringr)
library(car)
library(plyr)
require(xgboost)

#数值类型转化
for (i in 1:ncol(data1)){
  data1[,i]=as.numeric(data1[,i])
}

#XGBoost算法过程
require(xgboost)
## Loading required package: xgboost
require(methods)
require(plyr)
set.seed(123)
n <- nrow(data1)
index = sample(n,round(0.7*n))
train.xg = data1[index,]
test.xg = data1[-index,]


label <- as.matrix(train.xg['竞争力系数'])
data <- as.matrix(train.xg[,1:(ncol(data1)-1)])

label2 <- as.matrix(test.xg['竞争力系数'])
data2 <- as.matrix(test.xg[,1:(ncol(data1)-1)])

xgmat <- xgb.DMatrix(data, label = label, missing = -10000)
param <- list("objective" = "multi:softmax",
              "bst:eta" = 0.000000001,
              "gamma"=0.5,
              "bst:max_depth" = 6,
              "eval_metric" = "logloss","rmse","auc","error",
              "silent" = 0,
              "nthread"=16,
              "min_child_weight" =1.5,
              "max_delta_step"=1,
              "subsample"=0.5,
              "num_class"=5
)
nround =1000
bst = xgb.train(param, xgmat, nround )
res1 = predict(bst,data2)
table(res1,label2)
mean(res1 ==label2)


model <- xgb.dump(bst, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model
names <- dimnames(data.matrix(data1[,1:(ncol(data1)-1)]))[[2]]
# 计算特征重要性矩阵
importance_matrix <- xgb.importance(names, model = bst)
# 制图抓取最重要的因素变量
library(Ckmean.1d.dp)
xgb.plot.importance(importance_matrix[1:6,])
#做交叉验证
cv.res <- xgb.cv(data = data, label = label, max.depth = 6,eta = 0.0001, nround = 2, objective = "multi:softmax",num_class=5,nfold = 5)
cv.res
