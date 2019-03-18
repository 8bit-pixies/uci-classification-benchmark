#' get satimage

library(mlbench)
library(dplyr)
library(readr)
library(rsample)
library(recipes)

train_data <- read_delim("data/satimage_sat.trn", col_names=F, delim=" ")
test_data <- read_delim("data/satimage_sat.tst", col_names=F, delim=" ")

label <- "X37"

train_data %>% write_csv("clean_data/sat_train.csv")
test_data %>% write_csv("clean_data/sat_test.csv")

rec_obj <- recipe(X37 ~ ., data = train_data) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())


trained_rec <- prep(rec_obj, training = train_data)

train_data <- bake(trained_rec, newdata = train_data)
test_data  <- bake(trained_rec, newdata = test_data)

train_data %>% write_csv("clean_data/sat_train_scale.csv")
test_data %>% write_csv("clean_data/sat_test_scale.csv")
