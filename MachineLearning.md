---
title: "Final Project - Practical Machine Learning"
author: "Jesús Pérez"
date: "3/8/2021"
output:
  html_document:
    keep_md: yes
---



# Introduction

This project use data from accelerometers on the belt, forearm, arm, and dumbell of 6 research participants. The participants were each instructed to perform an exercise, and the result was classified either as properly performed (Class A) or in four common weightlifting mistakes (Classes B, C, D, and E). Source: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>. 

The aim of this project is to evaluate diffent Machine Learning Model using a **Train** set and apply the best model to predict the labels for another **Test** set observations.

# Preparation of the Data

## Reading Data

As shown below there are 19622 observations and 160 variables in the **Train** dataset, and 20 observations and the same number of varibles in the **Test** dataset.


```r
File1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
File2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

TrainData <- read.csv(File1)
TestData <- read.csv(File2)

dim(TrainData)
```

```
## [1] 19622   160
```

```r
dim(TestData)
```

```
## [1]  20 160
```

## Cleaning Data

There were a lot of dummy variables in the original datasets, and several columns did not have measurements for each observation. For this reason, the variables that contains missing values were removed. All rows containing summary statistics were removed as well. This reduced considerably the size of the datasets, especially the **Train** set.


```r
Train <- TrainData[, colSums(is.na(TrainData)) == 0]
Test <- TestData[, colSums(is.na(TestData)) == 0]

Train <- Train[, -c(1:7)]
Test <- Test[, -c(1:7)]

dim(Train); dim(Test)
```

```
## [1] 19622    86
```

```
## [1] 20 53
```

The **Train** set was further claened by removing the variables that are near-zero-variance. This equaled the number of variables in both datasets.


```r
NZV <- nearZeroVar(Train)
Train <- Train[, -NZV]
dim(Train)
```

```
## [1] 19622    53
```

## Correlation Analysis

The correlation among variables was analysed before proceeding to the modeling procedures.


```r
corMatrix <- cor(Train[, -53])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

![Correlation Matrix](MachineLearning_files/figure-html/Exploring the data-1.png)

The results showed that some of the variables were highly correlated (more than 0.9). Those variables were also removed.


```r
corr <- findCorrelation(corMatrix, cutoff = .90)
Train <- Train[, -corr]
Test <- Test[, -corr]
```

# Model Building

Before starting to fit the models, a *cross validation* was carried out to the the **Train** model. That dataset was splited in two, a new **Train** dataset which contained the 70% of the data and a **Validation** dataset with the remaining 30%.


```r
set.seed(4321)
inTrain <- createDataPartition(Train$classe, p = 0.7, list= FALSE)
Train <- Train[inTrain, ]
Validation <- Train[-inTrain, ]
```

In this way, three methods were applied to model the regressions (in the **Train** dataset) and the best one (higher accuracy) was used for the prediction in the **Test** set. The methods were: 

*Decision Tree
*Random Forests
*Generalized Boosted Model

For each one, a confusion Matrix was showed to better visualize the accuracy of the models.

# Decision Tree


```r
control <- trainControl(method="cv", number=3, verboseIter=F)
trees <- train(classe~., data=Train, method="rpart", trControl = control, tuneLength = 5)
fancyRpartPlot(trees$finalModel)
```

![](MachineLearning_files/figure-html/Trees-1.png)<!-- -->

**Validation**


```r
Prediction_trees <- predict(trees, Validation)
CM_trees <- confusionMatrix(Prediction_trees, factor(Validation$classe))
CM_trees
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1050  318  321  245  150
##          B   27  241   26   16  134
##          C   66  166  305   72  162
##          D   24   60   82  282   60
##          E    0    2    0   44  269
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5209          
##                  95% CI : (0.5055, 0.5362)
##     No Information Rate : 0.2831          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3769          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8997  0.30623  0.41553  0.42792  0.34710
## Specificity            0.6501  0.93913  0.86246  0.93474  0.98626
## Pos Pred Value         0.5038  0.54279  0.39559  0.55512  0.85397
## Neg Pred Value         0.9426  0.85155  0.87198  0.89568  0.86709
## Prevalence             0.2831  0.19093  0.17807  0.15987  0.18802
## Detection Rate         0.2547  0.05847  0.07399  0.06841  0.06526
## Detection Prevalence   0.5056  0.10771  0.18705  0.12324  0.07642
## Balanced Accuracy      0.7749  0.62268  0.63899  0.68133  0.66668
```

# Random forest


```r
controlRF <- trainControl(method="cv", number=3, verboseIter=F)
RandomForest <- train(classe~., data=Train, method="rf", trControl = controlRF)
RandomForest$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 23
## 
##         OOB estimate of  error rate: 0.66%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3901    3    1    0    1 0.001280082
## B   19 2632    7    0    0 0.009781791
## C    0   14 2371   11    0 0.010434057
## D    0    0   23 2224    5 0.012433393
## E    0    0    3    4 2518 0.002772277
```

**Validation**


```r
Prediction_RandomForest <- predict(RandomForest, Validation)
CM_RandomForest <- confusionMatrix(Prediction_RandomForest, factor(Validation$classe))
CM_RandomForest
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1167    0    0    0    0
##          B    0  787    0    0    0
##          C    0    0  734    0    0
##          D    0    0    0  659    0
##          E    0    0    0    0  775
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2831     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000    1.000
## Specificity            1.0000   1.0000   1.0000   1.0000    1.000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000    1.000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000    1.000
## Prevalence             0.2831   0.1909   0.1781   0.1599    0.188
## Detection Rate         0.2831   0.1909   0.1781   0.1599    0.188
## Detection Prevalence   0.2831   0.1909   0.1781   0.1599    0.188
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000    1.000
```

# Generalized Boosted Model


```r
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
GBM  <- train(classe ~ ., data=Train, method = "gbm", trControl = controlGBM, verbose = FALSE)
GBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 45 predictors of which 45 had non-zero influence.
```

**Validation**


```r
Prediction_GBM <- predict(GBM, newdata=Validation)
CM_GBM <- confusionMatrix(Prediction_GBM, factor(Validation$classe))
CM_GBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1156   20    0    0    1
##          B    6  747   16    2    5
##          C    5   19  706   15    6
##          D    0    1   11  639    3
##          E    0    0    1    3  760
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9723          
##                  95% CI : (0.9669, 0.9771)
##     No Information Rate : 0.2831          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.965           
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9906   0.9492   0.9619   0.9697   0.9806
## Specificity            0.9929   0.9913   0.9867   0.9957   0.9988
## Pos Pred Value         0.9822   0.9626   0.9401   0.9771   0.9948
## Neg Pred Value         0.9963   0.9880   0.9917   0.9942   0.9955
## Prevalence             0.2831   0.1909   0.1781   0.1599   0.1880
## Detection Rate         0.2804   0.1812   0.1713   0.1550   0.1844
## Detection Prevalence   0.2855   0.1883   0.1822   0.1587   0.1853
## Balanced Accuracy      0.9917   0.9702   0.9743   0.9827   0.9897
```

## Comparison

By comparing the accuracy rate values of the three models, the *Random Forest* displayed the higher accuracy. For this reason, this model was used on the **Test** data to predict the *classe* variable.


```r
Table <- data.frame(CM_trees$overall, CM_RandomForest$overall, CM_GBM$overall)
Table[-(5:7), ]
```

```
##               CM_trees.overall CM_RandomForest.overall CM_GBM.overall
## Accuracy             0.5208637               1.0000000      0.9723435
## Kappa                0.3768704               1.0000000      0.9650165
## AccuracyLower        0.5054816               0.9991055      0.9668691
## AccuracyUpper        0.5362162               1.0000000      0.9771339
```

# Prediction 

Finally, the Random Forest Model was used to predict the outcome of the **Test** set. These results will be used to answer the “*Course Project Prediction Quiz*”.


```r
Predict <- predict(RandomForest, Test)
Predict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
