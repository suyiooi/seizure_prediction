# Seizure Recurrence Prediction
# 
# Author  : Suyi Ooi
# Version : to run on server (Benji)
# Date    : Nov 2024

# --- Code Settings ------------------------------------------------------------

# number of outer folds 
NUM_OUTER_FOLDS <- 10

    
# --- Libraries & source functions ----------------------------------------------------------

library(dplyr)
library(ggplot2)

#ml/mlr3
library(mlr3) #v0.21.0 is on Benji
library(mlr3learners)
library(mlr3tuning)
library(mlr3filters)
library(mlr3fselect)
library(mlr3pipelines)
library(mlr3benchmark)
library(mlr3data)
library(paradox)
library(mlr3measures)
library(e1071)
library(xgboost)
library(ranger)
library(glmnet)
library(mlr3tuningspaces)

#harmonisation
library(neuroCombat)

#parallelisation
library(future)
library(future.apply)

#functions and hyperparameters
#setwd("/home/projects/suyi_structural_mri/seizure_prediction/")
source("functions.R")


# --- Data : load and clean ----------------------------------------------------

#load t1 (clin and imaging), and brainPAD
t1 <- read.csv("t1w.clin.image.scan.201.csv", header = TRUE)
brainage <- read.csv("PyBrainAge_Output_suyi_20240709.csv", header = TRUE)

t1$X = NULL
t1$sex[t1$sex == "nonbinary"] <- "male"

brainage <- subset(brainage, select = c("BrainAge","BrainPAD"))

#combine imaging and brainage
t1.data <- data.frame(t1,brainage)

#remove bendigo skyra and one subject
t1.dat <- t1.data %>% filter(batch != "bendigo_skyra",
                            subject != "sub-eh0703")

#make categorical variables factor
t1.dat <- t1.dat %>% dplyr::mutate(across(where(is.character), factor))

#reference levels
t1.dat$szrec1 <- relevel(t1.dat$szrec1, ref = "no") #predicting YES
t1.dat$sz_type <- relevel(t1.dat$sz_type, ref = "unc") 
t1.dat$mri <- relevel(t1.dat$mri, ref = "normal")
t1.dat$eeg <- relevel(t1.dat$eeg, ref = "normal")
t1.dat$risk_factors___3 <- relevel(t1.dat$risk_factors___3, ref = "no")

#preprocess - prefix subcortical features for grep
start_hat <- which(names(t1.dat) == "r_lat_nucleus")
end_hat <- which(names(t1.dat) == "l_hippo_whole")
hat_range <- start_hat:end_hat
colnames(t1.dat)[start_hat:end_hat] <- paste0("subcort_", colnames(t1.dat[start_hat:end_hat]))


# --- Set up MLR3 processes  ---------------------------------------------------

#filters

filter_methods <- list(
  "imp" = po("filter", filter = flt("importance"), filter.nfeat = 10),
  "pca" = po("pca") %>>% po("filter", filter = flt("variance"), filter.frac = 0.8),
  #"perm" = po("filter", filter = flt("permutation"), param_vals = list(filter.frac = 0.3)),
  "auc_filter" = po("filter", filter = flt("auc"), param_vals = list(filter.frac = 0.5)),
  "no_filter" = po("nop"))

#factor preprocessing
factor_pipeline = po("scale","center") %>>% po("removeconstants") %>>%
  po("encode", method = "one-hot",
     affect_columns = selector_cardinality_greater_than(2),
     id = "low_card_enc") %>>%
  po("encode", method = "treatment",
     affect_columns = selector_type("factor"), id = "binary_enc") 

#task

#define the task
task = TaskClassif$new(t1.dat, id = "szrec", target = "szrec1", positive = "yes")

#learners
learners <- list(lrn("classif.ranger",
                     id = "classif.ranger",
                     importance = "permutation",
                     predict_type = "prob"),
                 lrn("classif.xgboost",
                     id = "classif.xgboost",
                     predict_type = "prob"),
                 lrn("classif.glmnet",
                     id = "classif.glmnet",
                     predict_type = "prob"),
                 lrn("classif.svm",
                     id = "classif.svm",
                     type = "C-classification", kernel = "linear",
                     predict_type = "prob"))

# define measure
measure = msr("classif.auc")

#hyperparameter search strategy
tuner = mlr3tuning::tnr("random_search", batch_size = 10) #change depending on cores

#store results
train_sets <- list()
test_sets <- list()

results <- list()

clin_rr_train <- list()
clin_auc <- list()
clin_pred <- list()
clin_model <- list()

#rsmp strategy
outer_rsmp <- rsmp("custom")
inner_rsmp = rsmp("cv", folds = 10)

# --- Run Nested Cross-Validation ----------------------------------------------

#custom make 10 outer folds
set.seed(4)
folds <- stratified_folds(task, NUM_OUTER_FOLDS)

# outer fold
for (i in seq_along(folds)) {
  
  # Set up Train/Test for this Outer Fold ======================================
  message("OUTER PROGRESS:  Running Outer Fold #",i," of ", NUM_OUTER_FOLDS)
  
  test_set <- folds[[i]]
  train_set <- setdiff(task$row_ids, test_set)
  outer_rsmp$instantiate(task, list(train_set), list(test_set))
  
  train_sets[[i]] <- train_set #store these to check the shuffled indices
  test_sets[[i]] <- test_set
  
  # Apply combat on the training set ===========================================
  train_data <- t1.dat[train_set, ]
  test_data <- t1.dat[test_set, ]
  
  #extract feature lists
  train_df_list <- extract_featuregroup_list(train_data)
  test_df_list  <- extract_featuregroup_list(test_data)
  
  #biological covariates for combat
  covars.train = model.matrix(~age+factor(sex)+factor(szrec1), data=train_data)
  
  #batch
  batch.train = train_data$batch
  batch.test = test_data$batch 
  
  #apply combat function
  combat.train <- do_neuroCombat(train_df_list, batch.train, covars.train)
  
  #get dat.combat and estimates
  dat.combat.train.list <- lapply(combat.train, function(x) x$dat.combat)
  
  dat.combat.train <- get_dat_combated_df(dat.combat.train.list)
  
  #apply combat on test using the previously generated estimates
  cb.train.estimates <- lapply(combat.train, function(x) x$estimates)
  
  dat.combat.test.list <- do_neuroCombatFromTraining(
    test_df_list, batch.test, cb.train.estimates)
  
  dat.combat.test <- get_dat_combated_df(dat.combat.test.list)
  
  # Set up data & learners for inner folds =====================================
  #extract clinical factor variables
  t1.clin <- extract_clin_factors(train_data)
  t1.clin.test <- extract_clin_factors(test_data)
  
  #combine combated df with clinical factors
  t1.train.new <- data.frame(t1.clin, dat.combat.train)
  t1.test.new <- data.frame(t1.clin.test, dat.combat.test)
  
  #outer train and test tasks
  outer_train_task = TaskClassif$new(t1.train.new, id = "train", target = "szrec1")
  outer_test_task = TaskClassif$new(t1.test.new, id = "test", target = "szrec1")
  
  #make inner train task from outer train set
  inner_train_task = outer_train_task$clone()
  
  graph_learners = list()
  
  for (learner in learners) {
    
    #make graph learners with the filters piped
    
    for (filter in names(filter_methods)) {
      
      graph_name <- paste(filter, learner$clone()$id, sep = ".")
      graph <- factor_pipeline %>>% filter_methods[[filter]] %>>%
        learner$clone()
      graph_learner <- GraphLearner$new(graph)
      graph_learner$id <- graph_name
      graph_learners <- c(graph_learners,
                          list(GraphLearner$new(graph_learner)))
    }
  } #end learners 
  
  # Learner loop: hyperparameter tuning for each graph learner ================
  
  results[[i]] <- list()
  
  for (j in seq_along(graph_learners)){
    
    message("    LEARNER PROGRESS: running learner #",j," of ", length(graph_learners), " graph learners.")
    
    #run inner loop in parallel
    #future::plan("multisession", workers = 8) #uses workers are specified
    future::plan("sequential") #perhaps this avoids the col_roles error
    
    glrn <- graph_learners[[j]]   ### HERE ### something very funky ...
    
    results[[i]][[j]] <- future_lapply(list(glrn), process_each_learner, future.seed = TRUE)
        # nb. the function process_each_learner is now in functions.R
        # it implements the resampling/hyperparam tuning of the inner loop, 
        #  and finally applys the learner (using optimal hyperparams) to this fold.
    
  } #end graph_learners 
  
  #---clinical model---#
  
  #clin task train
  clin_task = outer_train_task$clone()
  clin_task$select(c("sex","age","mri",
                     "nocturnal_sz",
                     "risk_factors___2","risk_factors___3",
                     "risk_factors___9","sz_type","eeg"))
  
  #clin task test
  clin_test_task = outer_test_task$clone()
  clin_test_task$select(c("sex","age","mri",
                          "nocturnal_sz",
                          "risk_factors___2","risk_factors___3",
                          "risk_factors___9","sz_type","eeg"))
  
  #clin learner
  lrn_clin = lrn("classif.log_reg", predict_type = "prob")
  
  clin_glm = as_learner(factor_pipeline %>>% lrn_clin)
  
  instance_clin <- TuningInstanceBatchSingleCrit$new(
    task = clin_task,
    learner = clin_glm,
    resampling = rsmp("cv", folds = 10),
    measure = measure,
    terminator = trm("evals", n_evals = 10) # change this
  )
  
  clin_rr <- resample(clin_task, clin_glm, rsmp("cv", folds = 10),
                      store_models = FALSE)
  clin_rr_train[[i]] <- clin_rr$aggregate(measure)
  
  #train on clin training task
  clin_glm$train(clin_task)
  
  #predict on test
  pred_clin = clin_glm$predict(clin_test_task)
  
  #clin auc
  clin_performance = pred_clin$score(msr("classif.auc"))
  
  #stores clinical model results
  
  clin_auc[[i]] <- clin_performance
  clin_pred[[i]] <- list(pred_clin)
  clin_model[[i]] <- clin_glm
  
  
} # end outer

# --- Save models --------------------------------------------------------------

#saveRDS(results, file="results_111024.RData")
#saveRDS(results, file = "/home/projects/suyi_structural_mri/seizure_prediction/results_161024.RData")

#saveRDS(clin_pred, file = "clin_pred_111024.RData")
#saveRDS(clin_model, file = "clin_model_111024.RData")

#to open
#results <- readRDS("~/results_190924.RData")
#test_sets <- readRDS("~/test_sets_110224.RData")
#train_sets <- readRDS("~/train_sets_110224.RData")

