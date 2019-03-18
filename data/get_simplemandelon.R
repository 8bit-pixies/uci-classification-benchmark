#' get satimage

library(mlbench)
library(dplyr)
library(readr)
library(rsample)
library(recipes)

data <- read_csv("data/simple_mandelon.csv")
#test_data <- 

label <- "y"

set.seed(100)
train_test_split <- initial_split(data, prop = 0.7)
train_test_split

train_data <- training(train_test_split)
test_data <- testing(train_test_split) 

train_data %>% write_csv("clean_data/simplemandelon_train.csv")
test_data %>% write_csv("clean_data/simplemandelon_test.csv")

