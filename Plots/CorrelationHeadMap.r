setwd("C:/Users/user/Desktop/Fall 2017 Course Material/Data Mining/Homework/Project")
read <- read.csv("census-income.data_WithImputation_UsingRandomForest.csv", check.names = FALSE)
read <- subset(read, select = -c(education) )
categoricalColumns <- c("WorkClass", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "Class")
for(column in categoricalColumns) { read[[column]] <- as.integer(read[[column]]) }
corrMatrix <- round(cor(read),2)
corrMatrix[lower.tri(corrMatrix)] <- NA
library("ggplot2")
library("reshape2")
melted_corrMatrix <- melt(corrMatrix, na.rm = TRUE)
ggheatmap <- ggplot(melted_corrMatrix, aes(Var2, Var1, fill = value))+
    geom_tile(color = "white")+
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                         midpoint = 0, limit = c(-1,1), space = "Lab", 
                         name="Pearson Correlation") +
    theme_minimal()+ # minimal theme
    theme(axis.text.x = element_text(angle = 90, vjust = 0.7, 
                                     size = 9, hjust = 1))+
    coord_fixed()
ggheatmap + 
    geom_text(aes(Var2, Var1, label = ""), color = "black", size = 4) +
    theme(
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.ticks = element_blank(),
        legend.justification = c(1, 0),
        legend.position = c(0.6, 0.7),
        legend.direction = "horizontal")+
    guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                                 title.position = "top", title.hjust = 0.5))






