setwd("C:/Users/user/Desktop/Fall 2017 Course Material/Data Mining/Homework/Project")
data <- read.csv("census-income.data.csv", check.name = FALSE)
data[data == " ?"] <- NA
summary(data)
library(VIM)
dataNew <- kNN(data, c("WorkClass","occupation","native-country"), k=7)
dataWrite <- subset(dataNew, select = Age:Class)
write.csv(dataWrite, file="census-income.data_WithImputation_UsingKNNClutering.csv", row.names=FALSE)