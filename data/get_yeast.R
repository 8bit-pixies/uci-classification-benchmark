# yeast dataset
# install.packages("nclRcourses", repos="http://R-Forge.R-project.org")

library(tidyverse)
library(rsample)
library(recipes)

yeast <- read_delim("data/yeast.data", " ", col_names = F) %>%
  mutate_if(is.character, as.factor) 
names(yeast) <- c("name", "mcg", "gvh", "alm", "mit", "ert", "pox",
                  "vac", "nuc", "yeast")

yeast <- yeast %>%
  mutate(yeast=as.numeric(yeast)-1)

set.seed(100)
train_test_split <- initial_split(yeast, prop = 0.7)
train_test_split

train_dat <- training(train_test_split)
test_dat  <- testing(train_test_split) 

rec_obj <- recipe(yeast ~ ., data = train_dat) %>%
  step_meanimpute(all_numeric()) %>%
  step_modeimpute(all_predictors(), -all_numeric()) %>%
  step_dummy(all_predictors(), -all_numeric()) %>%
  step_nzv(all_predictors())

trained_rec <- prep(rec_obj, training = train_dat)

train_data <- bake(trained_rec, newdata = train_dat)
test_data  <- bake(trained_rec, newdata = test_dat)

train_data %>% write_csv("clean_data/yeast_train.csv")
test_data %>% write_csv("clean_data/yeast_test.csv")



rec_obj <- recipe(yeast ~ ., data = train_dat) %>%
  step_meanimpute(all_numeric()) %>%
  step_modeimpute(all_predictors(), -all_numeric()) %>%
  step_dummy(all_predictors(), -all_numeric()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

trained_rec <- prep(rec_obj, training = train_dat)

train_data <- bake(trained_rec, newdata = train_dat)
test_data  <- bake(trained_rec, newdata = test_dat)

train_data %>% write_csv("clean_data/yeast_train_scale.csv")
test_data %>% write_csv("clean_data/yeast_test_scale.csv")
