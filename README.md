# Prediction of Low Income Levels 

## From 1994 United States Census Data    
##### Team members: Traci Lim, Yi Luo, Bill Skinner
---
This markdown document is a summary of a fairly well-documented 10-page report of the entire machine learning process, which includes a detailed study of 5 different machine learning algorithms, justifying their impact and comparing their performance. 

The code used is based on the programming language R.

### Background and aims

Governments interested in engaging in affirmative action to benefit lower income populations may find it useful to have a model that predicts whether people with certain characteristics would have earned less than 50 thousand dollars per year in 1994. This model would enable them to input the same characteristics for people today to see whether the income level they would likely have had in 1994 would have been less than 50 thousand dollars per year which could be used to evaluate and substantiate any findings on social welfare, say, improvement in living conditions or individual growth in paygrade. The aim of this project is to provide such a model that predicts whether income in the United States is less than fifty thousand dollars per year.    

---

### Methodology 



![flow_chart](flow_chart.png)



We first used univariate and multivariate analysis to examine the dataset, revealing the class imbalance problem. Then, to prepare the data for model building, we created four separate data sets to evaluate the effectiveness of three different resampling approaches to solve the class imbalance problem. To reach the best predicting power, we experimented with several learning methods, and we also normalized numerical variables when appropriate for the method. Considering both the success metrics and running time, we narrowed down to 5 main methods: 

- Logistic Regression
- Linear Discriminant Analysis (LDA)
- Classification and Regression Trees (CART)
- Tree Boosting (C5.0)
- Stacking

The predicting ability of our model has the potential application to improve the efficiency of a social welfare system where the government needs to allocate subsidies to those in the greatest need.    

---

### Summary and Interpretation of Results 

The following table shows a list of the well-performing classifiers we evaluated with the best model in each method, or local optimal model, highlighted with a darker colour hue. All running times are for when the code was run on an Intel(R) Core(TM) i7-7700HQ Lenovo laptop, with 16GB of RAM.    

![model_sumary_table](model_sumary_table.png)

Models trained on non-resampled training sets have generally poor specificity scores because standard classification models (e.g. logistic regression, SVM, decision trees, nearest neighbors) treat all classes as equally important and thus tend to be biased towards the major class in imbalanced problems, producing a high sensitivity score (E. Burnaev, P. Erofeev, & A. Papanov, 2017). 

Models trained on resampled training sets produced a lower Kappa statistic and sensitivity, despite boosting specificity favourably, as compared to models trained on non-resampling training sets. Although resampling approaches have improved the prediction of the minority class, it comes with an undesirable trade-off in a drop in Kappa and sensitivity. This could be due to the nature of random resampling approaches. For over-sampling, the replication of minority class samples likely led to overfitting. Likewise, for under-sampling, the sample chosen may be biased, resulting in inaccurate results.  

Logistic regression (LR) outperforms linear discriminant analysis (LDA) significantly, in terms of the Kappa statistic. This could be because LR, unlike LDA, is relatively robust, and it does not require assumptions to be made regarding the distribution of the features, whereas LDA assumes features are normally distributed. Since not all of the features are normally distributed, the usage of LDA is theoretically wrong, as the assumptions are not completely met, despite applying a Box-cox transformation (Pohar, Blas, & Turk, 2004). Tree-based classifier rpart displays higher scores compared to LR and LDA because of its flexibility in handling categorical features, and requires minimal assumptions. The top contender for the best model is a tree-based boosting version of C5.0. Other than being able to detect the relevancy of features, it handles the multi-value features well and mitigates the over fitting problem by automated pruning. Our final tuned C5.0 a tree-based C5.0 method is accompanied with 15-trial boosting, where 15 separate decision trees or rulesets are combined to make predictions. Figure 16 in Appendix A lists the top 13 features it uses to classify samples from the training set. We picked the best model by examining 3 main metrics: Kappa statistic, sensitivity, and Area under ROC curve. The best performing model is the stacked model: rpart, LDA, and C5.0 (highlighted in yellow). We combined the predictions of these three individually-tuned models using logistic regression. Its outstanding scores can be credited to its smoothing nature and ability to highlight each base model where it performs best and discredit each base model where it performs poorly (Gorman, 2016).

---

### Conclusion and takeaways

We obtained our best model through the use of the technique of Stacking, in which we combined three models (LDA, CART, and C5.0), attaining a sensitivity of 94.37%. This result means that given the set of feature values (age, education level, etc.) our model would correctly classify about 94% of people who make less than fifty thousand dollars per year, although it would only correctly classify about 63% of people who make more than that. 

For a welfare system that should benefit those in greatest need, even at the expense of “wasting” some subsidies on the less deprived, our model could be an effective tool for predicting which people have lower income based only on a set of survey characteristics. 

There are some notable takeaways from this project. Having models trained on different resampling methods taught us that resampling a class imbalance dataset might not deliver results compatible with preliminary objectives. Although there is likely no algorithm that performs consistently well every time, understanding the strengths and weakness of each method still gives one an edge over randomly fitting a myriad of models and hoping for the best. Overall, we are pleased with our approach in optimizing predictive performance. Many other algorithms were attempted, but they were computationally slow to train and tune. Given excess time, we may look into implementing parallel processing in R to speed up some of the computationally expensive tasks, like tuning Support Vector Machines and Random Forest models. We could also perform a deeper analysis to justify the grouping of levels in categorical features, because the grouping was done intuitively without thorough justification.    

---

### References

Alberg, J. (2015, June 14). R, caret, and Parameter Tuning C5.0. Retrieved from Euclidean Technologies: http://www.euclidean.com/machine-learning-in-practice/2015/6/12/r-caretand-parameter-tuning-c50 

E. Burnaev, P. Erofeev, & A. Papanov. (2017, July 12). Influence of Resampling on Accuracy of Imbalanced Classification. Institute for Information Transmission Problems (Kharkevich Institute) RAS. 

Fridley, J. (2010, February 22). Tree models in R. Retrieved from http://plantecology.syr.edu/fridley/bio793/cart.html 

Gorman, B. (2016, December 27). A Kaggler's Guide to Model Stacking in Practice. Retrieved from http://blog.kaggle.com: http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-modelstacking-in-practice/ 

Kohavi, R., & Becker, B. (1996, May 1). UC Irvine Machine Learning Repository. Retrieved from https://archive.ics.uci.edu/ml/datasets/Adult 

Longadge, R., Dongre, S. S., & Malik, L. (2013, February). Class Imbalance Problem in Data Mining: Review. International Journal of Computer Science and Network. ML Wave. (2015, June 11). KAGGLE ENSEMBLING GUIDE. Retrieved from mlwave.com: https://mlwave.com/kaggle-ensembling-guide/ 

Movin, M., & Jagelid, M. (2017). A Comparison of Resampling Techniques to Handle the Class Imbalance Problem in Machine Learning- Conversion prediction of Spotify Users - A Case Study. 

Pearson, R. (2016, 4 12). The GoodmanKruskal package: Measuring association between categorical variables. Retrieved from cran.r-project.org: https://cran.rproject.org/web/packages/GoodmanKruskal/vignettes/GoodmanKruskal.html 

Pohar, M., Blas, M., & Turk, S. (2004). Comparison of Logistic Regression and Linear Discriminant Analysis: A Simulation Study. Metodološki zvezki, pp. 143-161. Rulequest Research. (2017, March). C5.0: An Informal Tutorial. Retrieved from www.rulequest.com: https://www.rulequest.com/see5-unix.html#WINNOWING 

Scibilia, B. (2015, March 30). How Could You Benefit from a Box-Cox Transformation? Retrieved from The Minitab Blog: http://blog.minitab.com/blog/applying-statistics-in-quality-projects/howcould-you-benefit-from-a-box-cox-transformation 

Standard Wisdom LLC. (2011, December 29). Confusion Matrix – Another Single Value Metric – Kappa Statistic. Retrieved from http://standardwisdom.com: http://standardwisdom.com/softwarejournal/2011/12/confusion-matrix-another-singlevalue-metric-kappa-statistic/ 

Therneau, T. M., & Atkinson, E. J. (2018, February 23). An Introduction to Recursive Partitioning Using the RPART Routines. Retrieved from https://cran.r-project.org: https://cran.rproject.org/web/packages/rpart/vignettes/longintro.pdf    

Wasikowski, M. (2009). Combating the Class Imbalance Problem in Small Sample Data Sets.    

Wu, X., Kumar, V., Quinlan, J., Ghosh, J., Yang, Q., Motoda, H. Steinberg, D. (2008, January). Top 10 algorithms in data mining. Knowledge and Information Systems, pp. 1–37.    