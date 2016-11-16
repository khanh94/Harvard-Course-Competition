setwd('/Users/khanh94/Documents/Kaggle/Harvard')
library(xgboost)
library(data.table)

train <- fread('train_predictors.txt')
test <- fread('test_predictors.txt')
labels <- fread('train_labels.txt')

train <- cbind(train, labels)
colnames(train)[103] <- 'Label'

target <- train$Label
train$Label <- NULL
model_xgb<- xgboost(data=as.matrix(train), 
                    label=as.matrix(target), 
                    objective="binary:logistic", 
                    booster="gbtree",
                    nrounds=1000, 
                    eta=0.01, 
                    max_depth=3, 
                    subsample=0.75, 
                    colsample_bytree=0.8, 
                    min_child_weight=1, 
                    eval_metric="error")

preds = rep(0, nrow(test))
preds <- preds + predict(model_xgb, as.matrix(test))

preds[preds > 0.5] = 1
preds[preds < 0.5] = 0

submission <- fread('sample_submission.txt')
final_sub <- data.frame(index = c(1:33149), label = preds)
write.csv(final_sub, file='final_sub.csv', row.names=FALSE)
