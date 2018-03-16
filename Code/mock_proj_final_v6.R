# Authors: Traci Lim, Willian Skinner, Yi Luo
# Date: 11/03/2018 (final edit)

# Dataset Description:
# -> This data set contains weighted census data extracted from the 1994 & 1995 current population surveys.
# -> The data contains demographic and employment related variables. 
# -> Insights? Purpose of predicting more/less than 50k income?

# SOME NOTES: 
# -> "# ->" indicate personal comments/intuition
#    "#" indicate the main objective of code

# -> ***Changelog.v4*** Date: 07/03/2018 
# -> Updated the arrangement of the code.
# -> Edited section 6.0 & 6.5: Correlation btw numerical features; Correlation btw categorical features
#    (All correlation analysis now done on training set: traindf)
# -> Removed SVM, GBM, RF (because of computationally slow running time)
# -> Removed OHE (because our best model is a tree-based method C5.0, and it handles categorical features well)
# -> Added some libraries in Importing Data section 1.0
# -> Added references (Website links) on some sections, to reference where i got the code from.
#    for sections with no references, i took the code from the R-package documentation. 
# -> Added Resampling Approaches section 7.0
# -> Added Visualising Results section 11.0, ROC curve visualisation
# -> Updated running times for some classifiers
# -> Updated Kappa scores
# -> Note: logit is short for logistic regression (i know they are not the same, definition wise)\

# -> ***Changelog.v5*** Date: 09/03/2018
# -> Updated the arrangement of the code.
# -> Edited Visualising data and Analyse data sections: changed dataset to traindf.unprocessed
# -> Added tree diagram for rpart for visualisation
# -> Added final results summary of C5.0 code for intepretation
# -> Added comments in Resampling Approaches section


# -> ***Changelog.v6*** Date: 11/03/2018
# -> Updated running times for each model


# 1.0 Importing Data ===========================================================================================
setwd("C:/Users/longwind48/Google Drive/Programming/MA429/Mock Project")

# Extracting column names from adult.names file
ci_names_S <- scan("data/adult.names", skip = 96, sep = ':', what = list(''))
ci_names_S <- unlist(ci_names_S)
ci_names_S <- ci_names_S[c(TRUE, FALSE)]
ci_names_S <- c(ci_names_S, 'annual-income')
summary(ci_names_S)

# Import .data file with column names
cis_df <- read.table("data/adult.data", header = FALSE, sep=",",
                     col.names = ci_names_S)

# Import .test file with column names
cis_df_test <- read.table("data/adult.test", skip = 1, header = FALSE, sep=",",
                     col.names = ci_names_S)
# -> Turns out the response variable has a dot on the levels
# -> Let's remove it.
levels(cis_df_test$annual.income)[levels(cis_df_test$annual.income)==" <=50K."] <- " <=50K"
levels(cis_df_test$annual.income)[levels(cis_df_test$annual.income)==" >50K."] <- " >50K"

# Find the proportion of split, for spliting it back later
prop_split <- as.numeric(dim(cis_df)[1]/(dim(cis_df)[1]+dim(cis_df_test)[1]))

# Merge 2 datasets
dataset <- rbind(cis_df, cis_df_test)
# -> We merge 2 datasets because it could possibly improve our pre-modelling analysis due to the increase in data points.

# Load the libraries
library(plyr); library(dplyr); library(psych); library(caret); library(car); library(C50); library(e1071); 
library(caTools); library(ggplot2)
library(randomForest); library(ROCR); library(MASS)
library(rpart); library(MASS); library(GoodmanKruskal); library(corrplot); library(caretEnsemble)
#devtools::install_github("collectivemedia/tictoc")
#devtools::install_github("easyGgplot2", "kassambara")
library(tictoc); library(easyGgplot2); library(ggthemes); library(PRROC); library(pROC); library(randomcoloR)

# 1.5 Spliting the merged dataset ============================================================================
# -> Data is split in 2 different sets: training and testing sets

# Shrink the dataset to train our models faster, 10%
# cutoff = round(0.1*nrow(dataset))
# dataset <- dataset[1:cutoff,]
# create a list of 70% of the rows in the original dataset we can use for training
set.seed(7)
test.index <- createDataPartition(dataset$annual.income, p=0.7, list=FALSE)
# select 30% of the data for testing
testdf.unprocessed <- dataset[-test.index,]
# use the remaining 70% of data for creating training dataset
traindf.unprocessed <- dataset[test.index,]


# 2.0 Analyse Data ============================================================================================

#View(traindf.unprocessed)
head(traindf.unprocessed)
dim(traindf.unprocessed)
# -> We have 34190 instances to work with and 15 features/attributes.
str(traindf.unprocessed)
# -> We want to predict the last feature (annual.income)
# -> This is a binary classification problem, since there are only 2 levels for our reponse variable.

# Look at how balanced the classes in feature (annual.income) are
percentage_cis <- prop.table(table(traindf.unprocessed$annual.income)) * 100
cbind(freq=table(traindf.unprocessed$annual.income), percentage=percentage_cis)
# -> We can see that 76% of instances have <=50k annual income.
# -> This is an indicator of a class imbalance problem.
# -> Different metrics should be used. Kappa? 

# Looking at what type of features do we have to work with
sapply(traindf.unprocessed, class)
# -> 6 features are numerical, 8 features are categorical

sapply(traindf.unprocessed, levels)
# -> features like native.country has 42 levels, makes more sense to reduce it because 'United-States' significantly
#    outweighs the rest of the levels
# -> We could also combine some of the low frequency levels in some catergorical features. WHY? i forgot the reason...
# -> Some of the levels in our categorical features are similar, 
#    like 'seperated' and 'divorced', 'local-gov' and 'state-gov', so many more.

# Let's take a look at the counts of each level in our categorical feature.
summary(traindf.unprocessed, maxsum = 42) 
# -> There are many levels with low frequency, this gives us an idea of what to combine and what not to combine.
# -> We can also take a look at the histograms.
# -> But i already look at them, and made the call to combine levels in some of the categorical features, under section 4


# 3.0 Visualize traindf.unprocessed ======================================================================================

# Discover whether a gender income gap exist
ggplot(traindf.unprocessed, aes(annual.income, fill = sex)) + 
    geom_bar(stat = "count", position = 'dodge')+
    theme_few() +
    xlab("annual.income") +
    ylab("count") +
    scale_fill_discrete(name = "sex") + 
    ggtitle("Gender and Income")

# Discover whether a racial income gap exist
ggplot(traindf.unprocessed, aes(annual.income, fill = race)) + 
    geom_bar(stat = "count", position = 'dodge')+
    theme_few() +
    xlab("annual.income") +
    ylab("Count") +
    scale_fill_discrete(name = "race") + 
    ggtitle("Race and Income")

# Shows the proportion of groups in the features earning more than 50k
prop.table(table(traindf.unprocessed$annual.income[traindf.unprocessed$race==' Black']))
prop.table(table(traindf.unprocessed$annual.income[traindf.unprocessed$race==' White']))
prop.table(table(traindf.unprocessed$annual.income[traindf.unprocessed$race==' Other']))
prop.table(table(traindf.unprocessed$annual.income[traindf.unprocessed$sex==' Male']))
prop.table(table(traindf.unprocessed$annual.income[traindf.unprocessed$sex==' Female']))

# 3.1 Create barplots for each categorical feature
par(mfrow=c(2,2))
par(las=2)
for (i in 1:length(traindf.unprocessed)) {
    if (class(traindf.unprocessed[,i])=="factor") {
        plot(traindf.unprocessed[,i], main=colnames(traindf.unprocessed)[i], cex.names=0.65, horiz=TRUE)
    }
}
# -> (workclass) There are levels which are low in frequency; i.e. Without-pay and Never-worked
#    Can possibly combine state-gov, local-gov, federal-gov
#    also, self-emp-not-inc and self-emp-inc.
# -> (education) There are levels which can be combined too, i.e. all the grades.
# -> (marital-status) Married-AF-spouse is basically non-existent
# -> (occupation) probably can group some low freq occupations into 'others'
# -> (relationship) possibly linked to marital-status
#    Can combine husband and wife to form a new category 'married'
# -> (race) can combine low frequency races to 'others'
# -> (sex) it is clear that there are more males than females.
# -> (native-country) most of the people are us citizens, 
#    probably not helpful if we were to consider all low frequency categories,
#    Could keep mexico, or change into continents.

# 3.2 Create boxplots for each numerical feature
par(mfrow=c(2,3))
par(cex=0.7, mai=c(0.3,0.8,0.5,1))
for (i in 1:length(traindf.unprocessed)) {
    if (class(traindf.unprocessed[,i])=="integer" | class(traindf.unprocessed[,i])=="numeric" ) {
        boxplot(traindf.unprocessed[,i]~traindf.unprocessed[,15], main=colnames(traindf.unprocessed)[i])
    }
}
# -> Should we deal with outliers?

# 3.3 Create Histograms and Kernel density plots for each numerical feature
par(mfrow=c(2,1))
par(cex=0.7, mai=c(0.3,0.8,0.5,1))
for (i in 1:length(traindf.unprocessed)) {
    if (class(traindf.unprocessed[,i])=="integer" | class(traindf.unprocessed[,i])=="numeric" ) {
        hist(traindf.unprocessed[,i], breaks=12, col="red", main=colnames(traindf.unprocessed)[i])
        plot(density(traindf.unprocessed[,i]), main=colnames(traindf.unprocessed)[i])
    }
}
# -> (age) There is a clear right skew, should probably try to make it normal, or not?

# 3.4 Visualising the relationship between each pair for numerical features
pairs.panels(traindf.unprocessed[,c(1,3,5,11:13)], gap=0, bg=c('red', 'blue')[traindf.unprocessed$annual.income], pch=2)

# Use qqplot to see how well 'age' variable fits a normal distribution
qqnorm(traindf.unprocessed$age); qqline(traindf.unprocessed$age)
# -> Visual inspection is usually unreliable, 
#    possible to use a significance test comparing the sample distribution to a normal one i
library(nortest)
ad.test(traindf.unprocessed$age)
cvm.test(traindf.unprocessed$age)
# -> For both Cramer-von Mises test for normality and Anderson-Darling test for normality, p values are <0.05,
#    This most likely imply that it is not normally distributed.
# -> Should we do a power transform to make it more normal?


# Understanding capital.gain and capital.loss
par(mfrow=c(1,2))
par(cex=0.7, mai=c(0.3,0.8,0.5,1))
boxplot(traindf.unprocessed$capital.gain[traindf.unprocessed$capital.gain>0]~traindf.unprocessed$annual.income[traindf.unprocessed$capital.gain>0])
# -> 
library(Hmisc)
describeBy(traindf.unprocessed$capital.gain, group = traindf.unprocessed$annual.income, mat = TRUE )
summary(traindf.unprocessed$annual.income)
summary(traindf.unprocessed$annual.income[traindf.unprocessed$capital.gain>0])
summary(traindf.unprocessed$annual.income[traindf.unprocessed$capital.gain==0])
describe(traindf.unprocessed$annual.income[traindf.unprocessed$capital.loss>0])
describe(traindf.unprocessed$annual.income[traindf.unprocessed$capital.loss==0])
# -> Looks like capital gain and captial loss are not reliable indicators of having an income >50k
# -> Out of 25% of instances with >50k, 6k has no capital investments 1.6k has cap gain



# 4.0 Cleaning Data ============================================================================================
# -> Cleaning the merged dataset to split it again later.
# -> After our univariate and multivariate analysis, we proceed to clean the merged dataset

# 4.1 Dealing with (native.country) feature
# Merge all levels with low frequency into one level 'Others'
others <- levels(dataset[,14])[-40]
dataset[,14] <- recode(dataset[,14], "c(others)='Others'")

# 4.2 Dealing with (race) feature
others_n <- levels(dataset$race)[1:2]
dataset$race <- recode(dataset$race, "c(others_n)=' Other'")

# 4.3 Dealing with (education) feature
others_e <- levels(dataset$education)[c(1:7,14)]
dataset$education <- recode(dataset$education, "c(others_e)='Dropout'")
others_e2 <- levels(dataset$education)[c(1:2)]
dataset$education <- recode(dataset$education, "c(others_e2)='Associates'")
others_e3 <- levels(dataset$education)[c(3,6)]
dataset$education <- recode(dataset$education, "c(others_e3)='HSGrad'")
others_e3 <- levels(dataset$education)[c(2,4)]
dataset$education <- recode(dataset$education, "c(others_e3)='PhD'")

# 4.4 Dealing with (workclass) feature
# -> “Never worked” and “Without-Pay” are small groups, we can combine them into 'Not-working' level
others_wc <- levels(dataset$workclass)[c(6,7)]
dataset$workclass <- recode(dataset$workclass, "c(others_wc)='Self-employed'")
others_wc1 <- levels(dataset$workclass)[c(4,7)]
dataset$workclass <- recode(dataset$workclass, "c(others_wc1)='Not-working'")
others_wc2 <- levels(dataset$workclass)[c(3,5)]
dataset$workclass <- recode(dataset$workclass, "c(others_wc2)='Non-federal-gov'")

# 4.5 Dealing with (occupation) feature
others_o <- levels(dataset$occupation)[c(4,6,7,8,15)]
dataset$occupation <- recode(dataset$occupation, "c(others_o)='Blue-collar'")
others_o1 <- levels(dataset$occupation)[c(5,6)]
dataset$occupation <- recode(dataset$occupation, "c(others_o1)='Service'")
others_o2 <- levels(dataset$occupation)[c(6,8)]
dataset$occupation <- recode(dataset$occupation, "c(others_o2)='Other-occ'")

# 4.6 Dealing with (marital.status) feature
others_ms <- levels(dataset$marital.status)[c(4,6)]
dataset$marital.status <- recode(dataset$marital.status, "c(others_ms)='Not-w-spouse'")
others_ms1 <- levels(dataset$marital.status)[c(2,3)]
dataset$marital.status <- recode(dataset$marital.status, "c(others_ms1)='Married'")

# 4.7 Dealing with (annual.income) feature
levels(dataset$annual.income) <- c("less", "more")

# -> Now, the there are lesser levels in some of the categorical features,
#    we can then proceed to analysis.

# 5.0 Spliting the merged dataset ============================================================================
# -> Data is split in 2 different sets: training and testing sets

# Shrink the dataset to train our models faster, 10%
# cutoff = round(0.1*nrow(dataset))
# dataset <- dataset[1:cutoff,]
# Use the same index we used to split earlier
test.index
# Select 30% of the data for testing
testdf <- dataset[-test.index,]
# Use the remaining 70% of data for creating training dataset
traindf <- dataset[test.index,]

# 6.0 Correlations between categorical variables ==============================================================

# -> Relationship and marital-status sounds correlated just by the names, let's check using a chisquared test
table(traindf$marital.status, traindf$relationship)
chisq.test(traindf$marital.status, traindf$relationship, correct=FALSE)
# -> A chi-squared value indicates a substantial relationship between two variables.
# -> We can reject the null hypothesis since our p-value is smaller than 0.05
#    and conclude that our variables are likely to have a significant relationship.

# Check for relationships between all categorical variables
subset <- c(2,4,6:10,14,15)
GKmatrix <- GKtauDataframe(traindf[,subset])
plot(GKmatrix)
# -> Tau value from relationship to marital.status is 0.59, indicating a strong association
# -> Good to see that among all categorical variables, only maritalstatus and rship have strong associations.
GKtau(traindf$marital.status, traindf$relationship)
# -> The Goodman-Kruskal tau measure: knowledge of marital.status is predictive of relationship, and similar otherwise.
# -> Reference: https://cran.r-project.org/web/packages/GoodmanKruskal/vignettes/GoodmanKruskal.html

# 6.5 Correlations between numerical variables ============================================================

# -> To show this, we combining correlogram with the significance test, 
# -> Reference: http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram
# Build a function to compute a matrix of p-values
cor.mtest <- function(mat, ...) {
    mat <- as.matrix(mat)
    n <- ncol(mat)
    p.mat<- matrix(NA, n, n)
    diag(p.mat) <- 0
    for (i in 1:(n - 1)) {
        for (j in (i + 1):n) {
            tmp <- cor.test(mat[, i], mat[, j], ...)
            p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
        }
    }
    colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
    p.mat
}
cor <- cor(traindf[,c(1,3,5,11,12,13)])
p.mat <- cor.mtest(traindf[,c(1,3,5,11,12,13)])

# Build a correlogram
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(cor, method="color", col=col(200),  
         type="upper", order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
         p.mat = p.mat, sig.level = 0.01, 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE)
# -> correlations with p-value > 0.05 are considered as insignificant, and crosses are added for those.
# -> Don't seem to have any correlation among numerical variables, which is a good thing.


# 7.0 Resampling Approaches =====================================================================================

table(traindf$annual.income)

percentage_cis_resampled <- prop.table(table(traindf.resampled$annual.income)) * 100
cbind(freq=table(traindf.resampled$annual.income), percentage=percentage_cis_resampled)

library(ROSE)
# Over-sampling
traindf.resampled.over <- ovun.sample(annual.income ~ ., data = traindf, method = "over",N = 52018, seed=7)$data
table(traindf.resampled.over$annual.income)

# Under-sampling
traindf.resampled.under <- ovun.sample(annual.income ~ ., data = traindf, method = "under",N = 16362, seed=7)$data
table(traindf.resampled.under$annual.income)

# Over-Under-Sampling
# -> Minority class is oversampled with replacement and majority class is undersampled without replacement.
traindf.resampled.both <- ovun.sample(annual.income ~ ., data = traindf, method = "both", p=0.5, N = 34190, seed=7)$data
table(traindf.resampled.both$annual.income)


# 8.0 Evaluate Algorithms =======================================================
library(caret)
stats <- function (data, lev = NULL, model = NULL)  {
    c(postResample(data[, "pred"], data[, "obs"]),
      Sens = sensitivity(data[, "pred"], data[, "obs"]),
      Spec = specificity(data[, "pred"], data[, "obs"]))
}

control <- trainControl(method="cv", number=10, summaryFunction = stats, classProbs = TRUE)
# -> The function trainControl can be used to specifiy the type of resampling,
#    in this case, 10-fold cross validation.
metric <- "Kappa" 
# control1 <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
# -> Kappa or Cohen’s Kappa is like classification accuracy, 
#    except that it is normalized at the baseline of random chance on your dataset. 
# -> A more useful measure to use on problems that have an imbalance in the classes 

# -> We know we have some skewed distributions, 
#    We can use a type of power transform to adjust and normalize these distributions. 
# -> A good choice will be Box-cox transform, by using caret's Preprocess parameter.
# -> Some algorithms, like tree-based algorithms are usually invariant to transforms.
# -> We will attempt to fit our model in the following combinations:
#    (Unenconded features/OHE features, no transformation/normalisation/Box-Cox transform)
# -> At each model, before we try a different classification algorithm, 
#    we will pick the best combination and attempt to tune to optimise the CV Kappa score.
                                                                                   

# LOGIT **********************************************************************************
set.seed(7)
fit.logit <- train(annual.income~.-fnlwgt, data=traindf, method="glm", family="binomial", 
                   metric=metric, trControl=control)

set.seed(7)
fit.logit5 <- train(annual.income~.-fnlwgt, data=traindf, method="glm", family="binomial",
                    preProcess = c('BoxCox'),
                    metric=metric, trControl=control)
# -> fit.logit5: 23.62 sec elapsed

set.seed(7)
fit.logit6 <- train(annual.income~.-fnlwgt, data=traindf.resampled.both, method="glm", family="binomial",
                    preProcess = c('BoxCox'),
                    metric=metric, trControl=control)
# -> fit.logit6: 21.43 sec elapsed
set.seed(7)
fit.logit7 <- train(annual.income~.-fnlwgt, data=traindf.resampled.over, method="glm", family="binomial",
                    preProcess = c('BoxCox'),
                    metric=metric, trControl=control)
# -> fit.logit7: 30.6 sec elapsed

set.seed(7)
fit.logit8 <- train(annual.income~.-fnlwgt, data=traindf.resampled.under, method="glm", family="binomial",
                    preProcess = c('BoxCox'),
                    metric=metric, trControl=control)
# -> fit.logit8: 9.76 sec elapsed


results.logit <- resamples(list(logit=fit.logit, logit5=fit.logit5, logit6=fit.logit6, logit7=fit.logit7, logit8=fit.logit8))
summary(results.logit)
dotplot(results.logit, metric='Kappa')

# Get true test kappa score
predictions.logit5 <- predict(fit.logit5, newdata = testdf)
predictions.logit5.prob <- predict(fit.logit5, newdata = testdf, type='prob')$less
predictions.logit6 <- predict(fit.logit6, newdata = testdf)
predictions.logit6.prob <- predict(fit.logit6, newdata = testdf, type='prob')$less
roc(testdf$annual.income, predictions.logit6.prob)
predictions.logit7 <- predict(fit.logit7, newdata = testdf)
predictions.logit7.prob <- predict(fit.logit7, newdata = testdf, type='prob')$less
roc(testdf$annual.income, predictions.logit7.prob)

predictions.logit8 <- predict(fit.logit8, newdata = testdf)
confusionMatrix(predictions.logit5, testdf$annual.income)
confusionMatrix(predictions.logit6, testdf$annual.income)
confusionMatrix(predictions.logit7, testdf$annual.income)
confusionMatrix(predictions.logit8, testdf$annual.income)

# -> OHE = Unencoded 
# -> Original > Resampled
# -> (Fitted on oversampled train set: fit.logit7) Kappa : 0.5561, Sensitivity : 0.7999, Specificity : 0.8485 
# -> (Fitted on undersampled train set: fit.logit8) Kappa : 0.5537, Sensitivity : 0.7979, Specificity : 0.8491 
# -> (Fitted on both over&undersampled train set: fit.logit8) Kappa : 0.5544, Sensitivity : 0.7998, Specificity : 0.8463 
# -> (Fitted on original train set: fit.logit5) Kappa : 0.5629, Sensitivity : 0.9318, Specificity : 0.5947 


# LDA *********************************************************************************************

set.seed(7)
fit.lda <- train(annual.income~.-fnlwgt, data=traindf, method='lda', preProcess = c('scale', 'center'),
                 metric=metric, trControl=control)
# -> fit.lda: 5.64 sec elapsed

set.seed(7)
fit.lda1 <- train(annual.income~.-fnlwgt, data=traindf, method='lda', preProcess = 'BoxCox',
                  metric=metric, trControl=control)
# -> fit.lda1: 12.48 sec elapsed

set.seed(7)
fit.lda2 <- train(annual.income~.-fnlwgt, data=traindf.resampled.over, method='lda', preProcess = 'BoxCox',
                 metric=metric, trControl=control)
# -> fit.lda2: 19.86 sec elapsed

set.seed(7)
fit.lda3 <- train(annual.income~.-fnlwgt, data=traindf.resampled.under, method='lda', preProcess = 'BoxCox',
                  metric=metric, trControl=control)
# -> fit.lda3: 6.34 sec elapsed

set.seed(7)
fit.lda4 <- train(annual.income~.-fnlwgt, data=traindf.resampled.both, method='lda', preProcess = 'BoxCox',
                  metric=metric, trControl=control)
# -> fit.lda4: 12.96 sec elapsed

results.lda <- resamples(list(lda=fit.lda, lda1=fit.lda1, lda2=fit.lda2, lda3=fit.lda3, lda4=fit.lda4))
summary(results.lda)
dotplot(results.lda, metric=metric)

# Get true test kappa score
predictions.lda <- predict(fit.lda, newdata = testdf)
predictions.lda1 <- predict(fit.lda1, newdata = testdf)
predictions.lda1.prob <- predict(fit.lda1, newdata = testdf, type='prob')$less
predictions.lda2 <- predict(fit.lda2, newdata = testdf)
predictions.lda3 <- predict(fit.lda3, newdata = testdf)
predictions.lda4 <- predict(fit.lda4, newdata = testdf)
confusionMatrix(predictions.lda, testdf$annual.income)
confusionMatrix(predictions.lda1, testdf$annual.income)
confusionMatrix(predictions.lda2, testdf$annual.income)
confusionMatrix(predictions.lda3, testdf$annual.income)
confusionMatrix(predictions.lda4, testdf$annual.income)
# -> OHE = Uncoded
# -> (Fitted on oversampled train set: fit.lda2) Kappa : 0.5128, Sensitivity : 0.7592, Specificity : 0.8642 
# -> (Fitted on undersampled train set: fit.lda3) Kappa : 0.5127, Sensitivity : 0.7569, Specificity : 0.8685 
# -> (Fitted on both over&undersampled train set: fit.lda4) Kappa : 0.5153, Sensitivity : 0.7596, Specificity : 0.8671 
# -> (Fitted on original train set: fit.lda1) Kappa : 0.5269, Sensitivity : 0.9306, Specificity : 0.5559 
# -> LOGIT(0.5629) > LDA(0.5269)


# TREES ************************************************************************************************
# TREES: Cart ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -> Short for Classification and Regression Trees 

tic('fit.cart')
set.seed(7)
fit.cart <- train(annual.income~.-fnlwgt, data=traindf, method="rpart", 
                    parms = list(split = "information"), #or 'information'
                   metric=metric, trControl=control, tuneLength = 10)
toc()
# -> fit.cart: 13.75 sec elapsed


tic('fit.cart1')
set.seed(7)
fit.cart1 <- train(annual.income~.-fnlwgt, data=traindf.resampled.over, method="rpart", 
                  parms = list(split = "information"), #or 'information'
                  metric=metric, trControl=control, tuneLength = 10)
toc()
# -> fit.cart1: 21.75 sec elapsed
tic('fit.cart2')
set.seed(7)
fit.cart2 <- train(annual.income~.-fnlwgt, data=traindf.resampled.under, method="rpart", 
                  parms = list(split = "information"), #or 'information'
                  metric=metric, trControl=control, tuneLength = 10)
toc()
# -> fit.cart2: 5.79 sec elapsed
tic('fit.cart3')
set.seed(7)
fit.cart3 <- train(annual.income~.-fnlwgt, data=traindf.resampled.both, method="rpart", 
                   parms = list(split = "information"), #or 'information'
                   metric=metric, trControl=control, tuneLength = 10)
# -> fit.cart3: 12.91 sec elapsed
toc()


results.cart <- resamples(list(cart=fit.cart, cart1=fit.cart1, cart2=fit.cart2, cart3=fit.cart3 ))
summary(results.cart)
dotplot(results.cart, metric=metric)
# -> The splitting index Information outperforms Gini
# Visualising Rpart 
library(rattle)
fancyRpartPlot(fit.cart$finalModel)

# Get true test kappa score
predictions.cart <- predict(fit.cart, newdata = testdf)
predictions.cart.prob <- predict(fit.cart, newdata = testdf, type='prob')$less
predictions.cart1 <- predict(fit.cart1, newdata = testdf)
predictions.cart2 <- predict(fit.cart2, newdata = testdf)
predictions.cart3 <- predict(fit.cart3, newdata = testdf)
confusionMatrix(predictions.cart, testdf$annual.income)
confusionMatrix(predictions.cart1, testdf$annual.income)
confusionMatrix(predictions.cart2, testdf$annual.income)
confusionMatrix(predictions.cart3, testdf$annual.income)
# -> OHE = Unencoded
# -> information > Gini
# -> (Fitted on oversampled train set: fit.cart1) Kappa : 0.5436, Sensitivity : 0.7894, Specificity : 0.8511 
# -> (Fitted on undersampled train set: fit.cart2) Kappa : 0.5454, Sensitivity : 0.7856, Specificity : 0.8614 
# -> (Fitted on both over&undersampled train set: fit.cart3) Kappa : 0.5542, Sensitivity : 0.7984, Specificity : 0.8488 
# -> (Fitted on original train set: fit.cart) Kappa : 0.5812, Sensitivity : 0.9388, Specificity : 0.6018 
# -> CART(0.5812) > LOGIT(0.5629) > LDA(0.5269)

# TREES: Boosting ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# C5.0*******************************************************************************************************
grid.c50 <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(5,10,15,20,25), .model="tree" )

set.seed(7)
fit.C5.0 <- train(annual.income~.-fnlwgt, data=traindf,
                  method="C5.0", tuneGrid=grid.c50, metric=metric, trControl=control, verbose = FALSE)
# -> fit.C5.0: 223.87 sec elapsed

# -> Best tuning parameters are trials = 15, model = tree and winnow = FALSE.

set.seed(7)
fit.C5.01 <- train(annual.income~.-fnlwgt, data=traindf.resampled.over,
                  method="C5.0", tuneGrid=grid.c50, metric=metric, trControl=control, verbose = FALSE)
# -> fit.C5.01: 489.61 sec elapsed

set.seed(7)
fit.C5.02 <- train(annual.income~.-fnlwgt, data=traindf.resampled.under,
                   method="C5.0", tuneGrid=grid.c50, metric=metric, trControl=control, verbose = FALSE)
# -> fit.C5.02: 85.45 sec elapsed

set.seed(7)
fit.C5.03 <- train(annual.income~.-fnlwgt, data=traindf.resampled.both,
                   method="C5.0", tuneGrid=grid.c50, metric=metric, trControl=control, verbose = FALSE)
# -> fit.C5.03: 237.53 sec elapsed

results.c50 <- resamples(list(c5.0=fit.C5.0, C5.01=fit.C5.01, C5.02=fit.C5.02, C5.03=fit.C5.03))
summary(results.c50)
dotplot(results.c50, metric=metric)

# Summary of best model 
summary(fit.C5.0$finalModel)

predictions.C50 <- predict(fit.C5.0, newdata = testdf)
predictions.C50.prob <- predict(fit.C5.0, newdata = testdf, type='prob')$less
predictions.C501 <- predict(fit.C5.01, newdata = testdf)
predictions.C502 <- predict(fit.C5.02, newdata = testdf)
predictions.C503 <- predict(fit.C5.03, newdata = testdf)
confusionMatrix(predictions.C50, testdf$annual.income)
confusionMatrix(predictions.C501, testdf$annual.income)
confusionMatrix(predictions.C502, testdf$annual.income)
confusionMatrix(predictions.C503, testdf$annual.income)

# -> Unencoded > OHE
# -> (Fitted on oversampled train set: fit.C5.01) Kappa : 0.5919, Sensitivity : 0.8442, Specificity : 0.8095 
# -> (Fitted on undersampled train set: fit.C5.02) Kappa : 0.5889, Sensitivity : 0.8102, Specificity : 0.8754 
# -> (Fitted on both over&undersampled train set: fit.C5.03) Kappa : 0.5919, Sensitivity : 0.8442, Specificity : 0.8095 
# -> (Fitted on original train set: fit.C5.0) Kappa : 0.6099, Sensitivity : 0.9451, Specificity : 0.6224 
# -> C5.0(0.6099) > CART(0.5812) > LOGIT(0.5629) > LDA(0.5269)


# 9.0 Compare algorithms =============================================================
results.all <- resamples(list(logit=fit.logit5, lda=fit.lda1, cart=fit.cart, C5.0=fit.C5.0)) 
summary(results.all)
# Estimate Skill on Validation Dataset
# create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. 
dotplot(results.all)
dotplot(results.all, metric=metric)
# -> C5.0 is the best classifier, gave the best coss-validation kappa score.
# -> In order to improve the kappa score even further, we will look into model ensembling in the next section


# summarize Best Model
print(fit.C5.0)


# 10.0 Stacking Algorithms ===============================================================================
# -> AKA Model Ensembling
# -> Combine different classifiers using model stacking
# -> In other words, combine the predictions of multiple caret models using the caretEnsemble package.
# -> reference: https://cran.r-project.org/web/packages/caretEnsemble/vignettes/caretEnsemble-intro.html

# Before stacking, we check correlations between predictions made by separate models, 
# -> If correlation is (< 0.75), stacking is more likely to be effective.
modelCor(results.all)
# -> Since logit is very 

# Specify the type of resampling, in this case, repeated 10-fold cross validation
trainControl <- trainControl(method="cv", number=10,
                             savePredictions=TRUE, classProbs=TRUE)

# Create a caretlist object (basically a list of models together)
tic('model_list_big2')
model_list_big2 <- caretList(
    annual.income~.-fnlwgt, data=traindf,
    trControl=trainControl,
    tuneList=list(
        cart<-caretModelSpec(method = 'rpart', parms = list(split = "information"), tuneLength=10),
        lda<-caretModelSpec(method="lda", preProcess= 'BoxCox'),
        c50<-caretModelSpec(method="C5.0", tuneGrid=data.frame(.trials = 15, .model='tree', .winnow=FALSE))
    )
)
toc()
# model_list_big2: 99.67 sec elapsed


# Visualise the results of the list of models created by the caretlist() command
results.stack <- resamples(model_list_big2)

summary(results.stack)
dotplot(results.stack)

# Let's check for correlation between the list of models (caretlist object).
# -> If the predictions for the sub-models were highly corrected (> 0:75),
#    then they would be making very similar predictions most of the time, reducing the benefit of combining the predictions.
modelCor(results.stack)
splom(results.stack)
# -> Now create a stacked model using a simple linear model using caretStack(), 
# -> caretStack allows us to use “meta-models” to ensemble collections of predictive models. 
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3,
                             savePredictions=TRUE, classProbs=TRUE)

set.seed(7)
fit.stack5 <- caretStack(model_list_big2, method="glm", metric=metric, trControl=stackControl)
set.seed(7)
fit.stack6 <- caretStack(model_list_big2, method="rpart", metric=metric, trControl=stackControl)


predictions.stack.glm5 <- predict(fit.stack5, newdata = testdf)
predictions.stack.glm5.prob <- predict(fit.stack5, newdata = testdf, type='prob')
predictions.stack.rpart6 <- predict(fit.stack6, newdata = testdf)
predictions.stack.rpart6.probs <- predict(fit.stack6, newdata = testdf, type='prob')


confusionMatrix(predictions.stack.glm5, testdf$annual.income)
confusionMatrix(predictions.stack.rpart6, testdf$annual.income)
# -> Kappa : 0.6246 (model_list_big2=(cart, lda, C5.0), rpart) **WINNER!**
# -> Kappa : 0.6129 (model_list_big2=(cart, lda, C5.0), glm)


# 11.0 Visualising Results =========================================================================================
# Create a ROC plot comparing performance of all models
colors <- randomColor(count = 10, hue = c("random"), luminosity = c("dark"))
roc1 <- roc(testdf$annual.income, predictions.stack.glm5.prob, col=colors[1], percent=TRUE, asp = NA,
            plot=TRUE, print.auc=TRUE, grid=TRUE, main="ROC comparison", print.auc.x=70, print.auc.y=80)
roc2 <- roc(testdf$annual.income, predictions.stack.rpart6.probs, plot=TRUE, add=TRUE, 
            percent=roc1$percent, col=colors[2], print.auc=TRUE, print.auc.x=70, print.auc.y=70)
roc3 <- roc(testdf$annual.income, predictions.C50.prob, plot=TRUE, add=TRUE, 
            percent=roc1$percent, col=colors[3], print.auc=TRUE, print.auc.x=70, print.auc.y=60)
roc4 <- roc(testdf$annual.income, predictions.cart.prob, plot=TRUE, add=TRUE, 
            percent=roc1$percent, col=colors[4], print.auc=TRUE, print.auc.x=70, print.auc.y=50)
roc5 <- roc(testdf$annual.income, predictions.lda1.prob, plot=TRUE, add=TRUE, 
            percent=roc1$percent, col=colors[5], print.auc=TRUE, print.auc.x=70, print.auc.y=40)
roc6 <- roc(testdf$annual.income, predictions.logit5.prob, plot=TRUE, add=TRUE, 
            percent=roc1$percent, col=colors[6], print.auc=TRUE, print.auc.x=70, print.auc.y=30)
legend("bottomright", legend=c("stack.glm", "stack.rpart", "C5.0", "CART", "LDA", "logistic"), col=c(colors[1:6]), lwd=2)

# 12.0 Conclusions ===================================================================================================
# -> STACK(0.6129) > C5.0(0.6099) > CART(0.5812) > LOGIT(0.5629) > LDA(0.5269)

# -> The kappa scores listed above are all test-kappa scores, and it was used with 100% of the data.
# -> The dotplots are all cv-kappa scores, 
#    quite frequently, some models loses out on cv-kappa scores but trumps on test-kappa scores.

# -> I have included the runtime of the best selected models in each algorithm.

# -> Looking at the 2 performance metrics, Kappa and Sensitivity, 
# -> we decided that stack.rpart (using classification trees to stack) is the best model. 
# -> based on Kappa statistic, sensitivity and area under roc curve score.


