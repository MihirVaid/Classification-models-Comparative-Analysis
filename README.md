# Classification-models-Comparative-Analysis
Predict students' dropout and academic success


##Abstract:

This machine learning project focuses on predicting student enrollment outcomes using five classification algorithms: Gaussian Naive Bayes, Support Vector Machine, Random Forest, K Nearest Neighbor, and Logistic Regression. The dataset contains student demographic information, educational background, admission grades, and academic performance in the first and second semesters. The goal is to classify students into three categories: "Dropout," "Enrolled," and "Graduate."

The project compares the performance of the algorithms in terms of accuracy, precision, recall, and F1-score to determine the most effective model.

The results of this project provide valuable insights into factors influencing student success or dropout rates. Educational institutions can use these findings to implement targeted interventions and support systems, ultimately improving student retention and success.

In summary, this machine learning project offers educational institutions a valuable tool to understand student enrollment outcomes and make informed decisions to support student success.


##Introduction

In this classification project, we explore the application of machine learning algorithms on the "studentsuccess.csv" dataset. The dataset contains various attributes related to students' characteristics and their academic performance. The goal of this project is to build a classification model that can predict the likelihood of student success based on the available features.
To achieve this, we performed exploratory data analysis to gain insights into the dataset, including examining the distribution of variables and exploring relationships between different attributes. We then performed feature selection to choose most important features, and split the dataset into training and testing sets.
Next, we employed different classification algorithms, such as Naive Bayes, Decision Trees, and Support Vector Machines, to train predictive models. We used appropriate evaluation metrics to assess the performance of these models, including accuracy, precision, recall, and F1-score. Additionally, we employed techniques like grid search to fine-tune the hyperparameters of the models, aiming to improve their predictive performance.

##Proposed Methodology

 




###Dataset:
 
Attribute Information of the dataset:
class 'pandas.core.frame.DataFrame'>
RangeIndex: 4424 entries, 0 to 4423
Data columns (total 37 columns):
 #   Column                                          Non-Null Count  Dtype 
---  ------                                          --------------  ----- 
 0   Marital status          	                    4424 non-null   int64 
 1   Application mode                            	4424 non-null   int64 
 2   Application order                           	4424 non-null   int64 
 3   Course                                      	4424 non-null   int64 
 4   Daytime/evening attendance 	                 	4424 non-null   int64 
 5   Previous qualification                      	4424 non-null   int64 
 6   Previous qualification (grade)              	4424 non-null   float64
 7   Nacionality                                 	4424 non-null   int64 
 8   Mother's qualification                      	4424 non-null   int64 
 9   Father's qualification                      	4424 non-null   int64 
 10  Mother's occupation   	                      4424 non-null   int64 
 11  Father's occupation                         	4424 non-null   int64 
 12  Admission grade                             	4424 non-null   float64
 13  Displaced                                   	4424 non-null   int64 
 14  Educational special needs                   	4424 non-null   int64 
 15  Debtor                        	              4424 non-null   int64 
 16  Tuition fees up to date                     	4424 non-null   int64 
 17  Gender                                      	4424 non-null   int64 
 18  Scholarship holder                          	4424 non-null   int64 
 19  Age at enrollment                               4424 non-null   int64 
 20  International                               	4424 non-null   int64 
 21  Curricular units 1st sem (credited)         	4424 non-null   int64 
 22  Curricular units 1st sem (enrolled)         	4424 non-null   int64 
 23  Curricular units 1st sem (evaluations)      	4424 non-null   int64 
 24  Curricular units 1st sem (approved)         	4424 non-null   int64 
 25  Curricular units 1st sem (grade)            	4424 non-null   float64
 26  Curricular units 1st sem (without evaluations)  4424 non-null   int64 
 27  Curricular units 2nd sem (credited)         	4424 non-null   int64 
 28  Curricular units 2nd sem (enrolled)         	4424 non-null   int64 
 29  Curricular units 2nd sem (evaluations)      	4424 non-null   int64 
 30  Curricular units 2nd sem (approved)         	4424 non-null   int64 
 31  Curricular units 2nd sem (grade)            	4424 non-null   float64
 32  Curricular units 2nd sem (without evaluations)  4424 non-null   int64 
 33  Unemployment rate                           	4424 non-null   float64
 34  Inflation rate                              	4424 non-null   float64
 35  GDP                                             4424 non-null   float64
 36  Target                                      	4424 non-null   object
dtypes: float64(7), int64(29), object(1)
memory usage: 1.2+ MB
None
 

##Heat map before feature selection
A heatmap is a graphical representation of data where values are encoded as colors. In the context of feature selection, a heatmap can be used to visualize the difference in feature importance or correlation before and after performing feature selection. 
Before feature selection, a heatmap can show the correlation between different features in the dataset. High correlation between features indicates a strong relationship, which can lead to multicollinearity issues and negatively impact the performance of the model. By visualizing the heatmap, we can identify highly correlated features that may be redundant or provide redundant information.

 

##Heat map after feature selection

After feature selection, the heatmap can show the reduced set of features and their corresponding importance or correlation. This can help us understand how the feature selection process has affected the dataset. Ideally, we would expect to see a reduction in the number of features and a focus on the most important and relevant ones.
•	Selected Features: ['Curricular units 2nd sem (approved)', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 1st sem (grade)', 'Admission grade', 'Curricular units 2nd sem (evaluations)', 'Curricular units 1st sem (evaluations)', 'Previous qualification (grade)', 'Age at enrollment', 'Tuition fees up to date']
 


 
##Why use GridSearchCV-
GridSearchCV is a technique used to find the best settings for a machine learning model. These settings are called hyperparameters and need to be specified before training the model. Examples of hyperparameters include the learning rate or the number of layers in a neural network.
GridSearchCV automates the process of trying out different combinations of hyperparameter values and evaluating how well the model performs with each combination. It does this by dividing the data into smaller parts and training the model on different combinations of these parts. This helps us understand how well the model would perform on unseen data.
The benefit of using GridSearchCV is that it saves time and effort by automatically exploring different hyperparameter combinations. It helps us find the best settings for our model without having to try them all manually.
 

##Gaussian Naive Bayes:
Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem and assumes that the features are independent and follow a Gaussian (normal) distribution. It is often used for text classification and spam filtering tasks. Despite its simplicity and strong assumptions, Gaussian Naive Bayes can be surprisingly effective in many real-world scenarios. It is computationally efficient and requires a relatively small amount of training data. However, it may struggle with complex relationships between features.

 

##Support Vector Machine (SVM):
Support Vector Machine is a powerful supervised learning algorithm used for both classification and regression tasks. SVM finds a hyperplane that best separates different classes by maximizing the margin between the classes. It is effective in high-dimensional spaces and can handle datasets with complex decision boundaries. SVM allows for the use of different kernel functions to transform the input data, enabling nonlinear classification. However, SVM can be computationally intensive and sensitive to the choice of hyperparameters.

 
##Random Forest:
Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. Each tree is trained on a random subset of the training data, and the final prediction is determined by aggregating the predictions of all the trees. Random Forest reduces the risk of overfitting and increases the model's generalization ability. It can handle a large number of input variables and works well with both categorical and numerical features. Random Forest is robust against outliers and noisy data. However, it may be challenging to interpret the results due to the complexity of the ensemble.
 


##K Nearest Neighbor (KNN): 
K Nearest Neighbor is a non-parametric classification algorithm that assigns a class to a sample based on the classes of its nearest neighbors. KNN uses distance metrics to determine the similarity between samples and classifies them based on the majority vote of their K nearest neighbors. KNN is simple to understand and implement and can handle multi-class classification problems. It is also robust to noisy data. However, KNN can be computationally expensive, especially with large datasets, as it requires calculating distances between all training samples.
 



##Logistic Regression: 
Logistic Regression is a widely used binary classification algorithm that models the probability of a sample belonging to a particular class. It estimates the coefficients of the input variables by minimizing the log-loss function using optimization techniques. Logistic Regression is interpretable and provides insights into the influence of each feature on the outcome. It can handle both categorical and numerical features and works well with large datasets. However, Logistic Regression assumes a linear relationship between the features and the log-odds of the outcome, which may limit its performance on complex nonlinear relationships.
 
 


##Comparison between models:
 
 


##Result and Discussion:
As per the above comparison the graph we shows information that  among all the selected classifiers Random Forest (Training Accuracy: 0.8728454365640012, Testing Accuracy: 0.7412429378531074) performs the best for the given dataset.

##Conclusion and Future Works
This machine learning project serves as a valuable tool for educational institutions to enhance their understanding of student enrollment outcomes and make informed decisions to support student success and retention.
In future the prediction of the models can be optimised further by using ensemble methods such as stacking or boosting. Combining the predictions of multiple classifiers can often lead to better results than using a single algorithm.
We can also consider incorporating additional external datasets that may provide valuable information related to student enrollment outcomes. This could include data on employment rates, income levels, or academic support programs to gain a more comprehensive understanding of the factors influencing enrollment outcomes.
This will help us to predict better and thus become a better resource for education monitoring.




##References

Research Papers:
Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
Han, J., Kamber, M., & Pei, J. (2011). Data mining: concepts and techniques. Morgan Kaufmann.
Duda, R. O., Hart, P. E., & Stork, D. G. (2012). Pattern classification. John Wiley & Sons.
Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.

Books:
Raschka, S., & Mirjalili, V. (2019). Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2. Packt Publishing.
Müller, A. C., & Guido, S. (2016). Introduction to Machine Learning with Python: A Guide for Data Scientists. O'Reilly Media.

Websites:
●	https://www.javatpoint.com/classification-algorithm-in-machine-learning
●	https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm
●	https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,both%20classification%20and%20regression%20problems
●	https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning
●	https://www.javatpoint.com/machine-learning-naive-bayes-classifier
