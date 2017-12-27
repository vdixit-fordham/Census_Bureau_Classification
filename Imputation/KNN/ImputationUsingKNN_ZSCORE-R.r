setwd("C:/Users/user/Desktop/Fall 2017 Course Material/Data Mining/Homework/Project")
data <- read.csv("census-income.data.csv", check.name = FALSE)
data[data == " ?"] <- NA
dataNormalized <- data
dataNormalized[, "Age"] <- as.data.frame( scale(data[, c("Age")] ))
dataNormalized[, "fnlwgt"] <- as.data.frame( scale(data[, c("fnlwgt")] ))
dataNormalized[, "education-num"] <- as.data.frame( scale(data[, c("education-num")] ))
dataNormalized[, "capital-gain"] <- as.data.frame( scale(data[, c("capital-gain")] ))
dataNormalized[, "capital-loss"] <- as.data.frame( scale(data[, c("capital-loss")] ))
dataNormalized[, "hours-per-week"] <- as.data.frame( scale(data[, c("hours-per-week")] ))
summary(data)
summary(dataNormalized)
library(VIM)
dataNormalizedNew <- kNN(dataNormalized, c("WorkClass","occupation","native-country"), k=7)
dataNew <- dataNormalizedNew
dataNew[, "Age"] <- data[, "Age"]
dataNew[, "fnlwgt"] <- data[, "fnlwgt"]
dataNew[, "education-num"] <- data[, "education-num"]
dataNew[, "capital-gain"] <- data[, "capital-gain"]
dataNew[, "capital-loss"] <- data[, "capital-loss"]
dataNew[, "hours-per-week"] <- data[, "hours-per-week"]
dataWrite <- subset(dataNew, select = Age:Class)
write.csv(dataWrite, file="census-income.data_zscore_knn_impute.csv", row.names=FALSE)