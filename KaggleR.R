data <- unzip("/Users/lilol/Downloads/train_data.csv.zip", list = TRUE)

library(readr)

#read data1.csv into data frame
df1 <- read_csv(unzip("/Users/lilol/Downloads/train_data.csv.zip", "train_data.csv"))
colnames(df1)

library(ggplot2)
library(scales)
library(ggeasy)

#filter by SN 

df_SN <- df1
df_SN <- df_SN[df_SN$SN_filter == 1, ]

write.csv(df_SN, file = "SN_filtered_data.csv")

                  
                  