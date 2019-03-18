# get letters dataset
library(mlbench)
library(dplyr)
library(readr)
library(recipes)

data("LetterRecognition")

export <- LetterRecognition %>%
  mutate(lettr = as.numeric(as.factor(lettr))-1) 

export %>% write_csv("data/letter.csv")

export <- read_csv("data/letter.csv")

set.seed(100)
train_test_split <- initial_split(export, prop = 0.7)
train_test_split

train_data <- training(train_test_split)
test_data <- testing(train_test_split) 
  
train_data %>% write_csv("clean_data/letter_train.csv")
test_data %>% write_csv("clean_data/letter_test.csv")


rec_obj <- recipe(lettr ~ ., data = train_data) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

trained_rec <- prep(rec_obj, training = train_data)

train_data <- bake(trained_rec, newdata = train_data)
test_data  <- bake(trained_rec, newdata = test_data)

train_data %>% write_csv("clean_data/letter_train_scale.csv")
test_data %>% write_csv("clean_data/letter_test_scale.csv")

