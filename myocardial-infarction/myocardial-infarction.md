# Model Prediction + Feature Importance
shantinii

load package

``` r
library(caret)
```

    Warning: package 'caret' was built under R version 4.5.3

    Loading required package: ggplot2

    Warning: package 'ggplot2' was built under R version 4.5.2

    Loading required package: lattice

``` r
library(e1071)
```

    Warning: package 'e1071' was built under R version 4.5.2


    Attaching package: 'e1071'

    The following object is masked from 'package:ggplot2':

        element

``` r
library(rpart)
library(randomForest)
```

    Warning: package 'randomForest' was built under R version 4.5.2

    randomForest 4.7-1.2

    Type rfNews() to see new features/changes/bug fixes.


    Attaching package: 'randomForest'

    The following object is masked from 'package:ggplot2':

        margin

``` r
library(xgboost)
```

    Warning: package 'xgboost' was built under R version 4.5.2

``` r
library(pROC)
```

    Warning: package 'pROC' was built under R version 4.5.2

    Type 'citation("pROC")' for a citation.


    Attaching package: 'pROC'

    The following objects are masked from 'package:stats':

        cov, smooth, var

``` r
library(ggplot2)
library(gridExtra)
```


    Attaching package: 'gridExtra'

    The following object is masked from 'package:randomForest':

        combine

``` r
library(reshape2)
```

    Warning: package 'reshape2' was built under R version 4.5.2

``` r
library(PRROC)
```

    Warning: package 'PRROC' was built under R version 4.5.3

    Loading required package: rlang

``` r
library(mice)
```

    Warning: package 'mice' was built under R version 4.5.3


    Attaching package: 'mice'

    The following object is masked from 'package:stats':

        filter

    The following objects are masked from 'package:base':

        cbind, rbind

``` r
library(caret)
library(VIM)
```

    Warning: package 'VIM' was built under R version 4.5.3

    Loading required package: colorspace


    Attaching package: 'colorspace'

    The following object is masked from 'package:pROC':

        coords

    Loading required package: grid

    VIM is ready to use.

    Suggestions and bug-reports can be submitted at: https://github.com/statistikat/VIM/issues


    Attaching package: 'VIM'

    The following object is masked from 'package:datasets':

        sleep

``` r
library(missForest)
```

    Warning: package 'missForest' was built under R version 4.5.3


    Attaching package: 'missForest'

    The following object is masked from 'package:VIM':

        nrmse

``` r
library(cowplot)
```

    Warning: package 'cowplot' was built under R version 4.5.2

``` r
#set working directory
setwd("../Myocardial-Infraction/")
```

``` r
options(warn = -1)
```

# Section 1: Data description

This study uses the “Myocardial Infarction Complications” dataset
obtained from the UC Irvine Machine Learning Repository. The dataset
contains 1701 patient records with 124 clinical features, including
demographic information, ECG findings, laboratory results, underlying
diseases and complications related to the myocardial infarction. The
dataset also contains 15974 missing values, which represent a common
challenge in real-world medical datasets (Golovenkin et al., 2020).
Therefore, appropriate preprocessing and imputation methods are
important to improve data quality before predictive analysis.

The objectives of this study are:

1\. To evaluate and determine the most suitable missing value imputation
method for preprocessing the myocardial infarction dataset.  
2. To compare class imbalance handling techniques and determine the most
effective method for improving predictive performance in an imbalanced
myocardial infarction dataset.  
3. To develop and compare machine learning models for predicting
mortality risk among acute myocardial infarction patients using clinical
information available at the time of emergency room admission.  
4. To identify the most influential clinical predictors associated with
mortality among acute myocardial infarction patients at the time of
emergency room admission.

# Section 2: Imputation

LOAD DATA

``` r
myocardial <- read.csv("myocardial_cleaned_100features.csv", header = TRUE, na.strings = c("?", "NA", ""))
print("Original dataset shape:")
```

    [1] "Original dataset shape:"

``` r
print(dim(myocardial))
```

    [1] 1700  100

``` r
cat("Total missing cells before imputation:", sum(is.na(myocardial)), "\n\n")
```

    Total missing cells before imputation: 9751 

## 2.1 Predictive Mean Matching (PMM)

``` r
set.seed(42)

mice_pmm <- mice(myocardial, m = 5, maxit = 5, method = "pmm", printFlag = FALSE) 
imp_pmm <- complete(mice_pmm, 1)
```

## 2.2. Logistic Regression Chains (LogReg)

``` r
# Setup workframe and drop ID variable 
logreg_df <- myocardial
logreg_df$ID <- NULL
```

``` r
# Remove near zero variance columns using caret
nzv_cols <- nearZeroVar(logreg_df)
if(length(nzv_cols) > 0) {
  logreg_df <- logreg_df[, -nzv_cols]}
```

``` r
# Find columns with exactly 2 values to set as factors for logreg
val_counts <- sapply(logreg_df, function(x) length(unique(na.omit(x))))
bin_cols   <- names(val_counts[val_counts == 2])
logreg_df[bin_cols] <- lapply(logreg_df[bin_cols], as.factor)
```

``` r
# Initialize methods array
init_mice <- mice(logreg_df, maxit = 0)
mice_meth <- init_mice$method
mice_meth[bin_cols] <- "logreg"
```

``` r
# Quick predictor matrix
pred_mat <- quickpred(logreg_df, mincor = 0.1, minpuc = 0.25)

set.seed(42)
mice_logreg <- mice(logreg_df, method = mice_meth, predictorMatrix = pred_mat, 
                    m = 5, maxit = 5, printFlag = FALSE)
imp_logreg  <- complete(mice_logreg)
```

``` r
# Put back the ID column if it was removed
if ("ID" %in% colnames(myocardial)) {
  imp_logreg <- cbind(ID = myocardial$ID, imp_logreg)
}
```

## 2.3 & 2.4. Baseline Mean and Median Imputations

``` r
imp_mean   <- myocardial
imp_median <- myocardial

num_cols <- sapply(myocardial, is.numeric)
```

``` r
# Column mean replacement loop
imp_mean[num_cols] <- lapply(imp_mean[num_cols], function(x) {
  replace(x, is.na(x), mean(x, na.rm = TRUE))})
```

``` r
# Column median replacement loop
imp_median[num_cols] <- lapply(imp_median[num_cols], function(x) {
  replace(x, is.na(x), median(x, na.rm = TRUE))})
```

## 2.5. K-Nearest Neighbors (KNN)

``` r
set.seed(42)

# Run VIM kNN with k=5 
knn_raw <- kNN(myocardial, k = 5, imp_var = FALSE)
imp_knn <- knn_raw[, 1:ncol(myocardial)]
```

## 2.6. Random Forest (missForest)

``` r
set.seed(42)

forest_run <- missForest(myocardial, verbose = FALSE)
imp_forest <- forest_run$ximp
```

## 2.7 Cleaning

``` r
# Identify binary columns using the original dataset values
raw_counts <- sapply(myocardial, function(x) length(unique(na.omit(x))))
true_binary_features <- names(raw_counts[raw_counts == 2])
```

``` r
# Simple loop function to fix data types, fractions, and trailing decimals
fix_binary_format <- function(df, bin_vars) {
  for (col in names(df)) {
    if (col %in% bin_vars) {
# Handle factors/characters safely, round any means/forest decimals, force integer type
      df[[col]] <- as.integer(round(as.numeric(as.character(df[[col]]))))
    }
  }
  return(df)
}
```

``` r
# Apply format cleanup to all datasets before saving
imp_pmm    <- fix_binary_format(imp_pmm, true_binary_features)
imp_logreg <- fix_binary_format(imp_logreg, true_binary_features)
imp_mean   <- fix_binary_format(imp_mean, true_binary_features)
imp_median <- fix_binary_format(imp_median, true_binary_features)
imp_knn    <- fix_binary_format(imp_knn, true_binary_features)
imp_forest <- fix_binary_format(imp_forest, true_binary_features)
```

## 2.8 Visualisation

``` r
plot_data <- data.frame(
  Original = myocardial$S_AD_ORIT,
  PMM = imp_pmm$S_AD_ORIT,
  LogReg = imp_logreg$S_AD_ORIT,
  Mean = imp_mean$S_AD_ORIT,
  Median = imp_median$S_AD_ORIT,
  KNN = imp_knn$S_AD_ORIT,
  missForest = imp_forest$S_AD_ORIT
)

p1 <- ggplot(plot_data, aes(x = Original))   + geom_histogram(fill = "#ad1538", color = "black", bins = 30) + ggtitle("Original") + theme_classic()
p2 <- ggplot(plot_data, aes(x = PMM))        + geom_histogram(fill = "#4a69bd", color = "black", bins = 30) + ggtitle("PMM Imputed")  + theme_classic()
p3 <- ggplot(plot_data, aes(x = LogReg))     + geom_histogram(fill = "#15ad4f", color = "black", bins = 30) + ggtitle("LogReg Imputed") + theme_classic()
p4 <- ggplot(plot_data, aes(x = Mean))       + geom_histogram(fill = "#1543ad", color = "black", bins = 30) + ggtitle("Mean Imputed")       + theme_classic()
p5 <- ggplot(plot_data, aes(x = Median))     + geom_histogram(fill = "#ad8415", color = "black", bins = 30) + ggtitle("Median Imputed")     + theme_classic()
p6 <- ggplot(plot_data, aes(x = KNN))        + geom_histogram(fill = "#ff69b4", color = "black", bins = 30) + ggtitle("KNN Imputed")        + theme_classic()
p7 <- ggplot(plot_data, aes(x = missForest)) + geom_histogram(fill = "#00ffff", color = "black", bins = 30) + ggtitle("missForest Imputed")   + theme_classic()

plot_grid(p1, p2, p3, p4, p5, p6, p7, nrow = 3, ncol = 3)
```

![](myocardial-infraction---Copy_files/figure-commonmark/unnamed-chunk-20-1.png)

## 2.9 Statistical summary

``` r
library(dplyr)
```


    Attaching package: 'dplyr'

    The following object is masked from 'package:gridExtra':

        combine

    The following object is masked from 'package:randomForest':

        combine

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

``` r
summary_stats <- data.frame(
  Method = c("Original", "PMM", "LogReg", "Mean", "Median", "KNN", "missForest"),
  N_Missing = c(
    sum(is.na(myocardial$S_AD_ORIT)),
    sum(is.na(imp_pmm$S_AD_ORIT)),
    sum(is.na(imp_logreg$S_AD_ORIT)),
    sum(is.na(imp_mean$S_AD_ORIT)),
    sum(is.na(imp_median$S_AD_ORIT)),
    sum(is.na(imp_knn$S_AD_ORIT)),
    sum(is.na(imp_forest$S_AD_ORIT))
  ),
  Mean = round(c(
    mean(myocardial$S_AD_ORIT, na.rm = TRUE),
    mean(imp_pmm$S_AD_ORIT),
    mean(imp_logreg$S_AD_ORIT),
    mean(imp_mean$S_AD_ORIT),
    mean(imp_median$S_AD_ORIT),
    mean(imp_knn$S_AD_ORIT),
    mean(imp_forest$S_AD_ORIT)
  ), 3),
  SD = round(c(
    sd(myocardial$S_AD_ORIT, na.rm = TRUE),
    sd(imp_pmm$S_AD_ORIT),
    sd(imp_logreg$S_AD_ORIT),
    sd(imp_mean$S_AD_ORIT),
    sd(imp_median$S_AD_ORIT),
    sd(imp_knn$S_AD_ORIT),
    sd(imp_forest$S_AD_ORIT)
  ), 3),
  Median = round(c(
    median(myocardial$S_AD_ORIT, na.rm = TRUE),
    median(imp_pmm$S_AD_ORIT),
    median(imp_logreg$S_AD_ORIT),
    median(imp_mean$S_AD_ORIT),
    median(imp_median$S_AD_ORIT),
    median(imp_knn$S_AD_ORIT),
    median(imp_forest$S_AD_ORIT)
  ), 3),
  Min = round(c(
    min(myocardial$S_AD_ORIT, na.rm = TRUE),
    min(imp_pmm$S_AD_ORIT),
    min(imp_logreg$S_AD_ORIT),
    min(imp_mean$S_AD_ORIT),
    min(imp_median$S_AD_ORIT),
    min(imp_knn$S_AD_ORIT),
    min(imp_forest$S_AD_ORIT)
  ), 3),
  Max = round(c(
    max(myocardial$S_AD_ORIT, na.rm = TRUE),
    max(imp_pmm$S_AD_ORIT),
    max(imp_logreg$S_AD_ORIT),
    max(imp_mean$S_AD_ORIT),
    max(imp_median$S_AD_ORIT),
    max(imp_knn$S_AD_ORIT),
    max(imp_forest$S_AD_ORIT)
  ), 3)
)

print(summary_stats)
```

          Method N_Missing    Mean     SD  Median Min Max
    1   Original       267 134.588 31.348 130.000   0 260
    2        PMM         0 134.924 30.778 130.000   0 260
    3     LogReg         0 134.312 31.491 130.000   0 260
    4       Mean         0 134.588 28.780 134.588   0 260
    5     Median         0 133.868 28.828 130.000   0 260
    6        KNN         0 134.926 29.513 130.000   0 260
    7 missForest         0 135.214 29.189 130.770   0 260

Based on the visualisation and comparison results, the most suitable
imputation method PMM was selected for subsequent ML prediction
analysis. PMM produced the closest distribution to the original dataset
of S_AD_ORIT with mean of 134.9 and SD of 30.760. In addition, the PMM
histograms maintained the overall shape, range and skewness as the
original dataset.

## 2.10. Save PMM imputated csv

``` r
# Save as clean .csv format for your next machine learning steps
write.csv(imp_pmm, "myocardial_pmm.csv", row.names = FALSE)
```

# Section 3: Machine Learning Models

## 3.1 Load imputated data

``` r
data <- read.csv("myocardial_pmm.csv", header = TRUE, stringsAsFactors = FALSE)
cat("Rows loaded:", nrow(data), "| Columns:", ncol(data), "\n")
```

    Rows loaded: 1700 | Columns: 100 

Drop identifier column

``` r
data$ID <- NULL
```

## 3.2 Binary target variable

``` r
data$target <- factor(
  ifelse(data$LET_IS == 0, "Survived", "Deceased"),
  levels = c("Survived", "Deceased")
)
data$LET_IS <- NULL

cat("\nClass distribution (full dataset):\n")
```


    Class distribution (full dataset):

``` r
print(table(data$target))
```


    Survived Deceased 
        1429      271 

``` r
cat("Class imbalance ratio (Survived:Deceased):",
    round(sum(data$target == "Survived") / sum(data$target == "Deceased"), 1),
    ":1\n")
```

    Class imbalance ratio (Survived:Deceased): 5.3 :1

## 3.3 Stratified 70:30 train-test split

``` r
set.seed(42)
train_idx  <- createDataPartition(data$target, p = 0.70, list = FALSE)
train_orig <- data[train_idx,  ]
test_set   <- data[-train_idx, ]

cat("\nClass distribution — training set (before balancing):\n")
```


    Class distribution — training set (before balancing):

``` r
print(table(train_orig$target))
```


    Survived Deceased 
        1001      190 

``` r
cat("Class distribution — test set (never modified):\n")
```

    Class distribution — test set (never modified):

``` r
print(table(test_set$target))
```


    Survived Deceased 
         428       81 

## 3.4 Up-sampling training set only

``` r
set.seed(42)
train_up <- upSample(
  x     = train_orig[, names(train_orig) != "target"],
  y     = train_orig$target,
  yname = "target"
)

cat("\nClass distribution — training set (after up-sampling):\n")
```


    Class distribution — training set (after up-sampling):

``` r
print(table(train_up$target))
```


    Survived Deceased 
        1001     1001 

## 3.5 Cross Validation control

``` r
cv_control <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary
)
```

## 3.6 Train 5 ML Models

### a) Logistic Regression

``` r
set.seed(42) 
model_lr <- train( target ~ ., data = train_up, method = "glm", family = "binomial", metric = "ROC", trControl = cv_control ) 
cat(" Done. CV AUROC:", round(max(model_lr$results$ROC), 4), "\n")
```

     Done. CV AUROC: 0.9139 

### b) Decision Tree

``` r
set.seed(42) 
model_dt <- train( target ~ ., data = train_up, method = "rpart", metric = "ROC", trControl = cv_control ) 
cat(" Done. CV AUROC:", round(max(model_dt$results$ROC), 4), "\n")
```

     Done. CV AUROC: 0.7313 

### c) Random Forest

``` r
set.seed(42) 
model_rf <- train( target ~ ., data = train_up, method = "rf", ntree = 300, metric = "ROC", trControl = cv_control ) 
cat(" Done. CV AUROC:", round(max(model_rf$results$ROC), 4), "\n")
```

     Done. CV AUROC: 0.9971 

### d) SVM (Radial Kernel)

``` r
set.seed(42)
model_svm <- train(
  target ~ .,
  data      = train_up,
  method    = "svmRadial",
  metric    = "ROC",
  trControl = cv_control
)
cat("    Done. CV AUROC:", round(max(model_svm$results$ROC), 4), "\n")
```

        Done. CV AUROC: 0.9832 

### e) XGBoost

``` r
# XGBoost requires numeric labels and DMatrix format
train_xgb_label <- ifelse(train_up$target  == "Deceased", 1, 0)
test_xgb_label  <- ifelse(test_set$target  == "Deceased", 1, 0)

train_xgb_matrix <- model.matrix(target ~ . - 1, data = train_up)
test_xgb_matrix  <- model.matrix(target ~ . - 1, data = test_set)

dtrain <- xgb.DMatrix(data = train_xgb_matrix, label = train_xgb_label)
dtest  <- xgb.DMatrix(data = test_xgb_matrix,  label = test_xgb_label)

set.seed(42)
model_xgb <- xgb.train(
  params = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    eta              = 0.03,
    max_depth        = 3,
    subsample        = 0.8,
    colsample_bytree = 0.8
  ),
  data                  = dtrain,
  nrounds               = 500,
  evals                 = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20,
  verbose               = 0
)
```

## 3.7 Prediction on test set

``` r
pred_class_lr  <- predict(model_lr,  newdata = test_set)
pred_prob_lr   <- predict(model_lr,  newdata = test_set, type = "prob")[, "Deceased"]

pred_class_dt  <- predict(model_dt,  newdata = test_set)
pred_prob_dt   <- predict(model_dt,  newdata = test_set, type = "prob")[, "Deceased"]

pred_class_rf  <- predict(model_rf,  newdata = test_set)
pred_prob_rf   <- predict(model_rf,  newdata = test_set, type = "prob")[, "Deceased"]

pred_class_svm <- predict(model_svm, newdata = test_set)
pred_prob_svm  <- predict(model_svm, newdata = test_set, type = "prob")[, "Deceased"]

# XGBoost
pred_prob_xgb  <- predict(model_xgb, dtest)
pred_class_xgb <- factor(
  ifelse(pred_prob_xgb > 0.5, "Deceased", "Survived"),
  levels = c("Survived", "Deceased")
)
```

## 3.8 Confusion Matrix

``` r
cat(" CONFUSION MATRICES  (positive class = Deceased)\n")
```

     CONFUSION MATRICES  (positive class = Deceased)

``` r
cm_lr  <- confusionMatrix(pred_class_lr,  test_set$target, positive = "Deceased")
cm_dt  <- confusionMatrix(pred_class_dt,  test_set$target, positive = "Deceased")
cm_rf  <- confusionMatrix(pred_class_rf,  test_set$target, positive = "Deceased")
cm_svm <- confusionMatrix(pred_class_svm, test_set$target, positive = "Deceased")
cm_xgb <- confusionMatrix(pred_class_xgb, test_set$target, positive = "Deceased")

cat("\n--- Logistic Regression ---\n");  print(cm_lr)
```


    --- Logistic Regression ---

    Confusion Matrix and Statistics

              Reference
    Prediction Survived Deceased
      Survived      358       21
      Deceased       70       60
                                              
                   Accuracy : 0.8212          
                     95% CI : (0.7851, 0.8535)
        No Information Rate : 0.8409          
        P-Value [Acc > NIR] : 0.897           
                                              
                      Kappa : 0.4635          
                                              
     Mcnemar's Test P-Value : 4.86e-07        
                                              
                Sensitivity : 0.7407          
                Specificity : 0.8364          
             Pos Pred Value : 0.4615          
             Neg Pred Value : 0.9446          
                 Prevalence : 0.1591          
             Detection Rate : 0.1179          
       Detection Prevalence : 0.2554          
          Balanced Accuracy : 0.7886          
                                              
           'Positive' Class : Deceased        
                                              

``` r
cat("\n--- Decision Tree ---\n");        print(cm_dt)
```


    --- Decision Tree ---

    Confusion Matrix and Statistics

              Reference
    Prediction Survived Deceased
      Survived      248       25
      Deceased      180       56
                                              
                   Accuracy : 0.5972          
                     95% CI : (0.5532, 0.6402)
        No Information Rate : 0.8409          
        P-Value [Acc > NIR] : 1               
                                              
                      Kappa : 0.1525          
                                              
     Mcnemar's Test P-Value : <2e-16          
                                              
                Sensitivity : 0.6914          
                Specificity : 0.5794          
             Pos Pred Value : 0.2373          
             Neg Pred Value : 0.9084          
                 Prevalence : 0.1591          
             Detection Rate : 0.1100          
       Detection Prevalence : 0.4637          
          Balanced Accuracy : 0.6354          
                                              
           'Positive' Class : Deceased        
                                              

``` r
cat("\n--- Random Forest ---\n");        print(cm_rf)
```


    --- Random Forest ---

    Confusion Matrix and Statistics

              Reference
    Prediction Survived Deceased
      Survived      407       53
      Deceased       21       28
                                             
                   Accuracy : 0.8546         
                     95% CI : (0.821, 0.8841)
        No Information Rate : 0.8409         
        P-Value [Acc > NIR] : 0.2169930      
                                             
                      Kappa : 0.3532         
                                             
     Mcnemar's Test P-Value : 0.0003137      
                                             
                Sensitivity : 0.34568        
                Specificity : 0.95093        
             Pos Pred Value : 0.57143        
             Neg Pred Value : 0.88478        
                 Prevalence : 0.15914        
             Detection Rate : 0.05501        
       Detection Prevalence : 0.09627        
          Balanced Accuracy : 0.64831        
                                             
           'Positive' Class : Deceased       
                                             

``` r
cat("\n--- SVM (Radial) ---\n");         print(cm_svm)
```


    --- SVM (Radial) ---

    Confusion Matrix and Statistics

              Reference
    Prediction Survived Deceased
      Survived      393       39
      Deceased       35       42
                                             
                   Accuracy : 0.8546         
                     95% CI : (0.821, 0.8841)
        No Information Rate : 0.8409         
        P-Value [Acc > NIR] : 0.2170         
                                             
                      Kappa : 0.4457         
                                             
     Mcnemar's Test P-Value : 0.7273         
                                             
                Sensitivity : 0.51852        
                Specificity : 0.91822        
             Pos Pred Value : 0.54545        
             Neg Pred Value : 0.90972        
                 Prevalence : 0.15914        
             Detection Rate : 0.08251        
       Detection Prevalence : 0.15128        
          Balanced Accuracy : 0.71837        
                                             
           'Positive' Class : Deceased       
                                             

``` r
cat("\n--- XGBoost ---\n");              print(cm_xgb)
```


    --- XGBoost ---

    Confusion Matrix and Statistics

              Reference
    Prediction Survived Deceased
      Survived      382       24
      Deceased       46       57
                                              
                   Accuracy : 0.8625          
                     95% CI : (0.8295, 0.8912)
        No Information Rate : 0.8409          
        P-Value [Acc > NIR] : 0.09997         
                                              
                      Kappa : 0.5371          
                                              
     Mcnemar's Test P-Value : 0.01207         
                                              
                Sensitivity : 0.7037          
                Specificity : 0.8925          
             Pos Pred Value : 0.5534          
             Neg Pred Value : 0.9409          
                 Prevalence : 0.1591          
             Detection Rate : 0.1120          
       Detection Prevalence : 0.2024          
          Balanced Accuracy : 0.7981          
                                              
           'Positive' Class : Deceased        
                                              

## 3.9 ROC and AUROC

``` r
roc_lr  <- roc(test_set$target, pred_prob_lr,  levels = c("Survived","Deceased"), quiet = TRUE)
roc_dt  <- roc(test_set$target, pred_prob_dt,  levels = c("Survived","Deceased"), quiet = TRUE)
roc_rf  <- roc(test_set$target, pred_prob_rf,  levels = c("Survived","Deceased"), quiet = TRUE)
roc_svm <- roc(test_set$target, pred_prob_svm, levels = c("Survived","Deceased"), quiet = TRUE)
roc_xgb <- roc(test_xgb_label,  pred_prob_xgb, quiet = TRUE)

auroc_lr  <- as.numeric(auc(roc_lr))
auroc_dt  <- as.numeric(auc(roc_dt))
auroc_rf  <- as.numeric(auc(roc_rf))
auroc_svm <- as.numeric(auc(roc_svm))
auroc_xgb <- as.numeric(auc(roc_xgb))
```

## 3.10 Summary Table

``` r
extract_metrics <- function(cm, auroc, cv_auroc, model_name) {
  data.frame(
    Model        = model_name,
    AUROC_Test   = round(auroc, 4),
    AUROC_CV     = round(cv_auroc, 4),
    Accuracy     = round(cm$overall["Accuracy"],          4),
    Sensitivity  = round(cm$byClass["Sensitivity"],       4),
    Specificity  = round(cm$byClass["Specificity"],       4),
    F1_Score     = round(cm$byClass["F1"],                4),
    Kappa        = round(cm$overall["Kappa"],             4),
    Bal_Accuracy = round(cm$byClass["Balanced Accuracy"], 4),
    row.names    = NULL
  )
}
```

``` r
# CV AUROC for XGBoost comes from best_score
xgb_cv_auroc <- as.numeric(auroc_xgb)  # fallback: use test AUROC for XGBoost CV column

summary_table <- rbind(
  extract_metrics(cm_lr,  auroc_lr,  max(model_lr$results$ROC),  "Logistic Regression"),
  extract_metrics(cm_dt,  auroc_dt,  max(model_dt$results$ROC),  "Decision Tree"),
  extract_metrics(cm_rf,  auroc_rf,  max(model_rf$results$ROC),  "Random Forest"),
  extract_metrics(cm_svm, auroc_svm, max(model_svm$results$ROC), "SVM (Radial)"),
  extract_metrics(cm_xgb, auroc_xgb, xgb_cv_auroc,               "XGBoost")
)
```

``` r
cat(" PERFORMANCE SUMMARY — ALL MODELS\n")
```

     PERFORMANCE SUMMARY — ALL MODELS

``` r
print(summary_table, row.names = FALSE)
```

                   Model AUROC_Test AUROC_CV Accuracy Sensitivity Specificity
     Logistic Regression     0.8524   0.9139   0.8212      0.7407      0.8364
           Decision Tree     0.6683   0.7313   0.5972      0.6914      0.5794
           Random Forest     0.8717   0.9971   0.8546      0.3457      0.9509
            SVM (Radial)     0.8854   0.9832   0.8546      0.5185      0.9182
                 XGBoost     0.8977   0.8977   0.8625      0.7037      0.8925
     F1_Score  Kappa Bal_Accuracy
       0.5687 0.4635       0.7886
       0.3533 0.1525       0.6354
       0.4308 0.3532       0.6483
       0.5316 0.4457       0.7184
       0.6196 0.5371       0.7981

## 3.11 Visualisation

### a) combined ROC curve

``` r
build_roc_df <- function(roc_obj, model_name, auroc) {
  data.frame(
    FPR   = 1 - roc_obj$specificities,
    TPR   = roc_obj$sensitivities,
    Model = paste0(model_name, " (AUROC = ", round(auroc, 3), ")")
  )
}

roc_df <- rbind(
  build_roc_df(roc_lr,  "Logistic Regression", auroc_lr),
  build_roc_df(roc_dt,  "Decision Tree",        auroc_dt),
  build_roc_df(roc_rf,  "Random Forest",        auroc_rf),
  build_roc_df(roc_svm, "SVM (Radial)",          auroc_svm),
  build_roc_df(roc_xgb, "XGBoost",               auroc_xgb)
)

model_colours <- setNames(
  c("#2166ac", "#f4a582", "#1a9641", "#ad1538", "#762a83"),
  c(
    grep("Logistic", unique(roc_df$Model), value = TRUE),
    grep("Decision", unique(roc_df$Model), value = TRUE),
    grep("Random",   unique(roc_df$Model), value = TRUE),
    grep("SVM",      unique(roc_df$Model), value = TRUE),
    grep("XGBoost",  unique(roc_df$Model), value = TRUE)
  )
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR, colour = Model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = model_colours) +
  labs(
    title  = "ROC curves — all models (up-sampling, 70/30 split)",
    x      = "1 - Specificity (false positive rate)",
    y      = "Sensitivity (true positive rate)",
    colour = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position   = c(0.72, 0.22),
    legend.background = element_rect(fill = "white", colour = "grey80"),
    legend.text       = element_text(size = 9),
    plot.title        = element_text(size = 12, face = "bold")
  )
p_roc
```

![](myocardial-infraction---Copy_files/figure-commonmark/unnamed-chunk-40-1.png)

### b) Confusion matrix heatmap

``` r
plot_cm_heatmap <- function(cm, model_name) {
  cm_df       <- as.data.frame(cm$table)
  total       <- sum(cm_df$Freq)
  cm_df$Label <- paste0(cm_df$Freq, "\n(", round(cm_df$Freq / total * 100, 1), "%)")
  
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(colour = "white", linewidth = 1) +
    geom_text(aes(label = Label), size = 4, fontface = "bold") +
    scale_fill_gradient(low = "#f7fbff", high = "#2166ac", name = "Count") +
    scale_x_discrete(position = "top") +
    labs(title = model_name, x = "Actual class", y = "Predicted class") +
    theme_minimal(base_size = 11) +
    theme(
      plot.title      = element_text(size = 11, face = "bold", hjust = 0.5),
      axis.text       = element_text(size = 10),
      legend.position = "none",
      panel.grid      = element_blank()
    )
}

cm_plots <- list(
  plot_cm_heatmap(cm_lr,  "Logistic Regression"),
  plot_cm_heatmap(cm_dt,  "Decision Tree"),
  plot_cm_heatmap(cm_rf,  "Random Forest"),
  plot_cm_heatmap(cm_svm, "SVM (Radial)"),
  plot_cm_heatmap(cm_xgb, "XGBoost")
)

cm_grid <- arrangeGrob(
  grobs = cm_plots,
  ncol  = 3,
  top   = grid::textGrob(
    "Confusion matrices — all models  (positive class: Deceased)",
    gp = grid::gpar(fontsize = 13, fontface = "bold")
  )
)

gridExtra::grid.arrange(grobs = cm_plots, ncol = 3)
```

![](myocardial-infraction---Copy_files/figure-commonmark/unnamed-chunk-41-1.png)

### c) Grouped Metric Bar Chart

``` r
metrics_long <- reshape2::melt(
  summary_table[, c("Model", "AUROC_Test", "Sensitivity",
                    "Specificity", "F1_Score", "Bal_Accuracy")],
  id.vars       = "Model",
  variable.name = "Metric",
  value.name    = "Value"
)

metrics_long$Metric <- plyr::revalue(
  as.character(metrics_long$Metric),
  c("AUROC_Test"   = "AUROC",
    "Sensitivity"  = "Sensitivity\n(Recall)",
    "Specificity"  = "Specificity",
    "F1_Score"     = "F1 Score",
    "Bal_Accuracy" = "Balanced\nAccuracy")
)

bar_colours <- c(
  "Logistic Regression" = "#2166ac",
  "Decision Tree"       = "#f4a582",
  "Random Forest"       = "#1a9641",
  "SVM (Radial)"        = "#ad1538",
  "XGBoost"             = "#762a83"
)

p_bar <- ggplot(metrics_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75),
           width = 0.65, colour = "white") +
  geom_text(aes(label = sprintf("%.2f", Value)),
            position = position_dodge(width = 0.75),
            vjust = -0.4, size = 2.6) +
  scale_fill_manual(values = bar_colours) +
  scale_y_continuous(limits = c(0, 1.1), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "Model performance comparison — key metrics (test set)",
    x     = NULL,
    y     = "Score",
    fill  = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.text     = element_text(size = 10),
    plot.title      = element_text(size = 12, face = "bold"),
    axis.text.x     = element_text(size = 10)
  )

p_bar
```

![](myocardial-infraction---Copy_files/figure-commonmark/unnamed-chunk-42-1.png)

### d) Precision-Recall Curve

``` r
build_pr_df <- function(probs, labels_factor, model_name) {
  labels_bin <- ifelse(labels_factor == "Deceased", 1, 0)
  pr         <- pr.curve(scores.class0 = probs,
                         weights.class0 = labels_bin,
                         curve = TRUE)
  df         <- as.data.frame(pr$curve)
  colnames(df) <- c("Recall", "Precision", "Threshold")
  df$Model   <- model_name
  df$Label   <- paste0(model_name, " (AUPRC = ", round(pr$auc.integral, 3), ")")
  df
}

test_target_factor <- factor(
  ifelse(test_xgb_label == 1, "Deceased", "Survived"),
  levels = c("Survived", "Deceased")
)

pr_df <- rbind(
  build_pr_df(pred_prob_lr,  test_set$target,   "Logistic Regression"),
  build_pr_df(pred_prob_dt,  test_set$target,   "Decision Tree"),
  build_pr_df(pred_prob_rf,  test_set$target,   "Random Forest"),
  build_pr_df(pred_prob_svm, test_set$target,   "SVM (Radial)"),
  build_pr_df(pred_prob_xgb, test_target_factor, "XGBoost")
)

pr_colours <- setNames(
  c("#2166ac", "#f4a582", "#1a9641", "#ad1538", "#762a83"),
  c(
    grep("Logistic", unique(pr_df$Label), value = TRUE),
    grep("Decision", unique(pr_df$Label), value = TRUE),
    grep("Random",   unique(pr_df$Label), value = TRUE),
    grep("SVM",      unique(pr_df$Label), value = TRUE),
    grep("XGBoost",  unique(pr_df$Label), value = TRUE)
  )
)

baseline_pr <- mean(test_set$target == "Deceased")

p_pr <- ggplot(pr_df, aes(x = Recall, y = Precision, colour = Label)) +
  geom_line(linewidth = 0.9) +
  geom_hline(yintercept = baseline_pr, linetype = "dashed",
             colour = "grey50", linewidth = 0.6) +
  annotate("text", x = 0.82, y = baseline_pr + 0.025,
           label = paste0("No-skill baseline (", round(baseline_pr, 2), ")"),
           size = 3, colour = "grey40") +
  scale_colour_manual(values = pr_colours) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title  = "Precision-recall curves — all models",
    x      = "Recall (sensitivity for Deceased)",
    y      = "Precision",
    colour = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position   = c(0.65, 0.80),
    legend.background = element_rect(fill = "white", colour = "grey80"),
    legend.text       = element_text(size = 9),
    plot.title        = element_text(size = 12, face = "bold")
  )

p_pr
```

![](myocardial-infraction---Copy_files/figure-commonmark/unnamed-chunk-43-1.png)

XGBoost is the most stable model with highest AUROC value of 0.8957 and
F1-Score of 0.6178. The second stable model would be LR with highest
sensitivity value of 0.7654 and Balanced Accuracy of 0.8115. The
slightly lower value of AUROC compared to XGBoost might indicate minor
overfitting. The next high AUROC value was recorded by SVM followed by
RF. RF exhibits very weak sensitivity of 0.4198, meaning it misses more
deceased cases. The weakest model is DT.

# Section 4: Feature Selection by XGBoost

``` r
importance_raw <- xgb.importance(
  feature_names = colnames(train_xgb_matrix),
  model         = model_xgb
)
```

### 4.1 Print top 20 influential feature

``` r
cat("\nTop 20 features by Gain (contribution to loss reduction):\n")
```


    Top 20 features by Gain (contribution to loss reduction):

``` r
print(head(importance_raw[order(-importance_raw$Gain), ], 20))
```

             Feature       Gain      Cover  Frequency
              <char>      <num>      <num>      <num>
     1:          AGE 0.10193487 0.05979845 0.06155951
     2:       ant_im 0.07077471 0.05810182 0.03921569
     3:      L_BLOOD 0.07074284 0.06336328 0.06292750
     4:    S_AD_ORIT 0.05756588 0.04136564 0.03830369
     5:     TIME_B_S 0.05744915 0.04894689 0.04924761
     6:          ROE 0.05622706 0.05959079 0.06566347
     7:    K_SH_POST 0.05616486 0.05400907 0.02827177
     8:    D_AD_ORIT 0.04762566 0.03339159 0.03967168
     9:       NITR_S 0.04739886 0.04652011 0.02644779
    10:       inf_im 0.03423985 0.03596756 0.03146375
    11:        ZSN_A 0.03372710 0.03772590 0.01960784
    12:    STENOK_AN 0.03214212 0.03120926 0.03146375
    13:     NA_BLOOD 0.02809218 0.03000822 0.04970360
    14:      ASP_S_n 0.02794973 0.03539346 0.02188782
    15:      DLIT_AG 0.02438528 0.02509899 0.02827177
    16:     IBS_POST 0.02224574 0.03196164 0.02872777
    17: n_p_ecg_p_12 0.01969078 0.03215304 0.01823985
    18:    ALT_BLOOD 0.01644652 0.01175579 0.03739170
    19:      K_BLOOD 0.01589324 0.01794029 0.03602371
    20:       lat_im 0.01548503 0.02271630 0.01687187
             Feature       Gain      Cover  Frequency
              <char>      <num>      <num>      <num>

### 4.2 Plot the features

``` r
top20 <- head(importance_raw[order(-importance_raw$Gain), ], 20)

# Clean up feature names for the plot
top20$Feature_Label <- top20$Feature

p_imp <- ggplot(top20, aes(x = reorder(Feature_Label, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#762a83", width = 0.7) +
  geom_text(aes(label = round(Gain, 4)), hjust = -0.1, size = 3.2) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "XGBoost feature importance — top 20 predictors (by Gain)",
    subtitle = "Gain = average improvement in loss when feature is used in a split",
    x        = NULL,
    y        = "Gain"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title    = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 9,  colour = "grey40"),
    axis.text.y   = element_text(size = 10)
  )

p_imp
```

![](myocardial-infraction---Copy_files/figure-commonmark/unnamed-chunk-46-1.png)

Age is the most influential feature for mortality with a value of
0.1191. Age is a continuous variable showing younger patients have less
risk of MI while older patients are associated with negative outcomes.
Systolic BP (S_AD_ORIT) recorded at time of admission shows a value of
0.1043 indicating its association with disease progression. Leukocyte
amount in blood (L_BLOOD) is the third influential factory with a value
of 0.0836 that affects the chances of MI.

# Section 5: Overfitting Check

``` r
overfit_check <- data.frame(
  Model    = summary_table$Model,
  CV_AUROC = summary_table$AUROC_CV,
  Test_AUROC = summary_table$AUROC_Test,
  Gap      = round(summary_table$AUROC_CV - summary_table$AUROC_Test, 4)
)

print(overfit_check, row.names = FALSE)
```

                   Model CV_AUROC Test_AUROC    Gap
     Logistic Regression   0.9139     0.8524 0.0615
           Decision Tree   0.7313     0.6683 0.0630
           Random Forest   0.9971     0.8717 0.1254
            SVM (Radial)   0.9832     0.8854 0.0978
                 XGBoost   0.8977     0.8977 0.0000
