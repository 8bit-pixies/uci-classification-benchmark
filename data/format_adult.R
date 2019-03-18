# adult preprocess
#' we will do a quick clean of adult to convert to numeric, mainly just one hot encoding and fixing missings?
library(rsample)
library(recipes)
library(tidyverse)
adult <- read_csv("data/adult.csv", na=c("?", "", "NA")) %>%
  mutate(income = as.numeric(as.factor(income))-1) %>%
  mutate_if(is.character,as.factor)

set.seed(100)
train_test_split <- initial_split(adult, prop = 0.7)
train_test_split

train_adult <- training(train_test_split)
test_adult  <- testing(train_test_split) 

rec_obj <- recipe(income ~ ., data = train_adult) %>%
  step_meanimpute(all_numeric()) %>%
  step_modeimpute(all_predictors(), -all_numeric()) %>%
  step_dummy(all_predictors(), -all_numeric()) 

trained_rec <- prep(rec_obj, training = train_adult)

train_data <- bake(trained_rec, newdata = train_adult)
test_data  <- bake(trained_rec, newdata = test_adult)

train_data %>% write_csv("clean_data/adult_train.csv")
test_data %>% write_csv("clean_data/adult_test.csv")


