#' get covtype
#' following info, first 11340 for training
#' 3780 for validation
#' testing is ignored for speed purposes

library(mlbench)
library(dplyr)
library(readr)
#library(rsample)
library(recipes)

covtype <- read_csv("data/covtype.data", col_names=F) %>%
  mutate(X55 = X55-1)
# response is X55

train_data <- covtype[1:11340,]
test_data <- covtype[11341:15120,]

train_data %>% write_csv("clean_data/covtype_train.csv")
test_data %>% write_csv("clean_data/covtype_test.csv")


rec_obj <- recipe(X55 ~ ., data = train_data) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  step_nzv(all_predictors())


trained_rec <- prep(rec_obj, training = train_data)

train_data <- bake(trained_rec, newdata = train_data)
test_data  <- bake(trained_rec, newdata = test_data)

train_data %>% write_csv("clean_data/covtype_train_scale.csv")
test_data %>% write_csv("clean_data/covtype_test_scale.csv")