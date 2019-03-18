# get mandelon

library(mlbench)
library(dplyr)
library(readr)
library(rsample)
library(recipes)

mandelon <- read_delim("data/madelon_train.data", col_names=F, delim=" ")
y <- read_delim("data/madelon_train.labels", col_names=F, delim=" ")
mandelon$y <- (y$X1+1)/2

mandelon <- mandelon %>%
  mutate_all(as.numeric)

set.seed(100)
train_test_split <- initial_split(mandelon, prop = 0.7)
train_test_split

train_data <- training(train_test_split)
test_data <- testing(train_test_split) 

train_data %>% write_csv("clean_data/mandelon_train.csv")
test_data %>% write_csv("clean_data/mandelon_test.csv")


rec_obj <- recipe(y ~ ., data = train_data) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

trained_rec <- prep(rec_obj, training = train_data)

train_data <- bake(trained_rec, newdata = train_data)
test_data  <- bake(trained_rec, newdata = test_data)

train_data %>% write_csv("clean_data/mandelon_train_scale.csv")
test_data %>% write_csv("clean_data/mandelon_test_scale.csv")