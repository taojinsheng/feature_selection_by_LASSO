#!/usr/bin/env Rscript
# CopyRight AnchorDx,All rights reserved CopyRight Jinsheng Tao
# <jinsheng_tao@anchordx.com>

args <- commandArgs(T)

library(glmnet)
library(sampling)
library(methods)
library(doParallel)
registerDoParallel(cores=10)

feature_selection <- function(train_x, train_y, sampling_times = 50) {
    beta_cal = data.frame(feature_name = colnames(train_x))
    train_y_sort = data.frame(rank = seq(length(train_y)), label = train_y)
    train_y_sort = train_y_sort[order(train_y), ]
    train_x = train_x[train_y_sort$rank, ]
    train_y = train_y_sort$label
    for (i in seq_len(sampling_times)) {
        set_myseed = as.integer(paste(1017, i, sep = ""))
        set.seed(set_myseed)
        # sampling sub1 = sample(nrow(train_x), as.integer(nrow(train_x)*0.75), replace =
        # F)
        sub1 = strata(train_y_sort, stratanames = "label", size = as.integer(c(table(train_y)[1] * 
            0.75, table(train_y)[2] * 0.75)), method = "srswor")
        train_x_sample = train_x[sub1$ID_unit, ]
        train_y_sample = train_y[sub1$ID_unit]
        
        # tuning parameter by 10 fold cv glmnet(train_x_sample, train_y_sample,
        # family='binomial',alpha=1);stop()
        fit_cv = cv.glmnet(train_x_sample, train_y_sample, family = "binomial", alpha = 1, 
            nfolds = 10, type.measure = "auc", parallel = T)  # run the model and get the parameter lambda.1se
        stopImplicitCluster()

        # fit model
        fit_model = glmnet(train_x_sample, train_y_sample, family = "binomial", alpha = 1, 
            lambda = fit_cv$lambda.1se)  # model
        
        feature_important = as.data.frame(as.matrix(fit_model$beta))
        feature_important = feature_important[colnames(train_x), ]
        beta_cal = cbind(beta_cal, feature_important)
        colnames(beta_cal)[i + 1] = paste("time_", i, sep = "")
        
        
    }
    return(beta_cal)
}

# 输入
train_x = read.delim(args[1], as.is = T, row.names = 1, check.names = F)

## find and replace with row mean for NA value
#ind<-which(is.na(train_x), arr.ind=TRUE)
#train_x[ind] <- rowMeans(train_x,  na.rm = TRUE)[ind[,1]]

train_x = train_x[rowSums(train_x != 0) >= 30, ]  #keep rows with >=30 non-zero observations

#na.idx <- apply(train_x,1,function(x){return(any(is.na(x)))})
#print(train_x[na.idx,])

print(dim(train_x))
train_x = t(train_x)  # transpose to make sample as row and feature as column

# train_x = train_x/max(train_x)
print(train_x[1:10, 1:10])
response = read.delim(args[2], as.is = T, check.names = F)
colnames(response) = c("sample", "group")

# df = data.frame(sample=rownames(train_x)) df = merge(df,response,by='sample')
response = response[match(rownames(train_x), response$sample), ]  #reorder sample order by matrix sample order
train_y = response$group

print(class(train_x))
print(class(train_y))
print(table(train_y))
# output = feature_selection(data.matrix(train_x),train_y,sampling_times = 500)
output = feature_selection(train_x, train_y, sampling_times = 500)
write.table(output, file = args[3], sep = "\t", quote = F, row.names = F)
