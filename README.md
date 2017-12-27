# Census_Bureau_Classification
This project requires to explore supervised learning algorithms on a real world data-set, and write a report explaining our experimental results.

1 Introduction :-
This project requires you to explore supervised learning algorithms on a real world dataset, and write a report explaining your experimental results. I used python to implement this project, the only requirement is that our program be able to interpret the data format specified below, and be able to classify instances and produce interesting statistics such as accuracy, false positive rate,
false negative rate, etc.

2 Algorithm :-
Our algorithm should be based on the supervised learning algorithms - LR, KNN, DT, SVM, NB, RF. Usually a straight forward implementation of one method will not lead to satisfactory performance. Our algorithm can be a combination of methods and should incorporate one or more
data mining techniques when the situation arises. These techniques include (and certainly not limited to):
Handling imbalanced dataset
Proper imputation methods for missing values
Different treatment of various type of features: continuous, discrete, categorical, etc.

3 Data :-
You'll be examining the behavior of your classification algorithm on a dataset from the UCI ma-chine learning lab. The dataset is represented in a standard format, consisting of 3 files. The first file, census-income.names, describes the categories and features of the dataset. It also has some empirical results for your reference. The other two files are census-income.data and census-income.test, containing the actual data instances, formatted at one instance per line. 
The data we will be examining was extracted from the census bureau database. Each instance contains an individual's educational, demographic and family information. Prediction task is to determine whether a person makes over 50K a year. We will use census-income.data to train our classifier and use census-income.test to evaluate the performance of our learning algorithm.
