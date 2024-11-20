# Seizure Recurrence Prediction
# 
# Author  : Suyi Ooi
# Version : to run on server (Benji)
# Date    : Nov 2024

# --- Code Settings ------------------------------------------------------------

  # outer for testing different algorithms-filter combos (learners)
NUM_OUTER_FOLDS <- 3  # 10 # for real run
 
 # inner for finding hyperparameters
NUM_INNER_FOLDS <- 5  # 10 # for real run

NUM_HYPERPARAM_LEVELS <- 20 # ?Do more for real. This controls the gridsearch, and is the number
                            # of distinct levels tried for each hyperparameter

NUM_HYPERPARAM_EVALS <- 20 # Do many more for real run. This controls the terminator. Total number of hyperparam sets
                           # currently doesn't do anything, since using trm("none")
    
CONST_RAND_SEED <- 4       # no variability between testing

GRID_SEARCH_BATCH_SIZE <- 20  # number of parameter sets to run in parallel during gridsearch

# --- Libraries & source functions ----------------------------------------------------------

library(dplyr)
library(ggplot2)
library(patchwork)

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

# for visualizations
library(mlr3viz)  

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
  ### DV - consider making nfeat / filter.frac tunable hyperparameters
filter_methods <- list(
  "no_filter" = po("nop"), 
  "imp" = po("filter", filter = flt("importance"), filter.nfeat = 10),
  "pca" = po("pca") %>>% po("filter", filter = flt("variance"), filter.frac = 0.8),
  "auc_filter" = po("filter", filter = flt("auc"), param_vals = list(filter.frac = 0.5))
  
  #"perm" = po("filter", filter = flt("permutation"), param_vals = list(filter.frac = 0.3)),
  )

#factor preprocessing
factor_pipeline = po("scale","center") %>>% 
                  po("removeconstants") %>>%
                  po("encode", method = "one-hot",
                     affect_columns = selector_cardinality_greater_than(2),
                     id = "low_card_enc") %>>%
                  po("encode", method = "treatment",
                     affect_columns = selector_type("factor"), id = "binary_enc") 

#task - define the task
task = TaskClassif$new(t1.dat, id = "szrec", target = "szrec1", positive = "yes")

#learners
learners <- list( lrn("classif.svm",
                      id = "classif.svm",
                      type = "C-classification", kernel = "linear",  # DV- consider using RBF when nfeat < nobs
                      predict_type = "prob"),
                  
                 lrn("classif.ranger",
                     id = "classif.ranger",
                     importance = "permutation",
                     predict_type = "prob"),
                 
                 lrn("classif.xgboost",
                     id = "classif.xgboost",
                     predict_type = "prob"),
                 
                 lrn("classif.glmnet",
                     id = "classif.glmnet",
                     predict_type = "prob")
                 )

# define measure
measure = msr("classif.auc")

#hyperparameter search strategy
#tuner = mlr3tuning::tnr("random_search", batch_size = 10) #change depending on cores
# DV - suggest trying Bayesian optimizatino or annealing tuning algorithsm
#    - a grid search would be robust but likely slow, esp when many hyperparams
#    - consider Hyperband  optimizer

# here is an attempt with a grid search
tuner = mlr3tuning::tnr("grid_search",
                        resolution = NUM_HYPERPARAM_LEVELS,  # number of values to try for each hyperparam
                        batch_size = GRID_SEARCH_BATCH_SIZE  # number of levels to run in parallel
                        ) 

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
inner_rsmp = rsmp("cv", folds = NUM_INNER_FOLDS)

# --- Run Nested Cross-Validation ----------------------------------------------

#custom make 10 outer folds
set.seed(CONST_RAND_SEED)
folds <- stratified_folds(task, NUM_OUTER_FOLDS)

### Outer Folds loop ###
for (i in seq_along(folds)) {
  message("OUTER FOLD LOOP:  Running Fold #",i," of ", NUM_OUTER_FOLDS)
  
  # Set up Train/Test for this Outer Fold ======================================
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
  
  # Set up data for inner folds ================================================
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
  
  # Set up learners (with filters) for inner folds =============================
  
  graph_learners = list()
  for (learner in learners) {
    
    #make graph learners with the filters piped
    for (filter in names(filter_methods)) {
      graph_name <- paste(filter, learner$clone()$id, sep = ".")
      graph <- factor_pipeline %>>% 
               filter_methods[[filter]] %>>%
               learner$clone()
      graph_learner <- GraphLearner$new(graph)
      graph_learner$id <- graph_name
      graph_learners <- c(graph_learners,
                          list(GraphLearner$new(graph_learner)))
    }
  } #end learners 
  
  # Learner loop: train-test each algorithm on this fold ================
  
  results[[i]] <- list()
  
    # DV - replace the below for loop with lapply (better for future_lapply switch, if needed)
  #results[[i]] <- lapply(graph_learners, 
  #                       process_each_learner, 
  #                       num_hyperparam_evals = NUM_HYPERPARAM_EVALS )
  
  ### Learners loop (4 algorithms x 4 filter strategies)
  for (j in seq_along(graph_learners)){
    
    glrn <- graph_learners[[j]]  
    message("    LEARNER LOOP: running learner #",j," of ", length(graph_learners), " <", glrn$id, ">, for outer fold #", i, ".")
    
    #run inner loop in parallel
    #future::plan("multisession", workers = 8) #uses workers are specified
    #future::plan("sequential") #perhaps this avoids the col_roles error
    # DV - remove parallelization for testing. 
    #      Consider replacing with parallelization of outer loop. 
    
    
  
    # DV - remove parallelization for testing
    # results[[i]][[j]] <- future_lapply(list(glrn), process_each_learner, future.seed = TRUE)
    
    #fallback   # DV - don't encapsulate/no fallback for testing
    #glrn$encapsulate(method = "evaluate", fallback = lrn("classif.log_reg", predict_type = "prob")) #mlr3 v0.21.0 requires this but the fall-back keeps kicking in
    #glrn$encapsulate <- c(train = "evaluate", predict = "evaluate") #this works on mlr3 v0.20.0 without the fallback kicking in
    #glrn$fallback = lrn("classif.log_reg", predict_type = "prob")
    
    #define hyperparameter space to explore
    param_set <- create_param_sets(glrn)
    
    #tuning instance
    instance <- TuningInstanceBatchSingleCrit$new(
      task = inner_train_task,
      learner = glrn,
      resampling = inner_rsmp,
      measure = measure,
      search_space = param_set,
        # grid search stops itself, so no terminator needed
      terminator = trm("none")
      
      #terminator = trm("evals", n_evals = NUM_HYPERPARAM_EVALS ) 
        # Option: use trm("combo") to combine termination after n_evals OR after an amount of time (eg 1 minute) ie trm("run_time")
    )
    
    # run tuner to find optimized hyperparameters, 
    #  using inner_rsmp folds (ie NUM_INNER_FOLDS )
    tuner$optimize(instance)
    # TO DO - check for stability of hyperparameters over instances
    
    #assign tuned hyperparameters to learner
    glrn$param_set$values = instance$result_learner_param_vals
    
    #retrain inner using final aggregated param_set$values 
    rr_inner <- resample(inner_train_task, glrn,
                         inner_rsmp,
                         store_models = FALSE)
    inner_auc <- rr_inner$aggregate(measure)
    
    # -- Outer loop train/test evaluation ------------------------------------
    
    glrn$train(outer_train_task)
    outer_prediction <- glrn$predict(outer_test_task)
    outer_auc <- outer_prediction$score(measure)
    
    #store results for each glrn
    results[[i]][[j]] <- list(
      learner = glrn$id,
      inner_performance = inner_auc,
      outer_performance = outer_auc,
      best_params = instance$result_learner_param_vals,
      outer_prediction = outer_prediction,
      model = glrn
    )
    
    #REPORTING - display performance and parameters for each outerfold_x_learner-filter
      # task parameters
    
      # visualize the task
    p<-list()
    p[[1]] <- autoplot(outer_train_task) + ggtitle("Outer Train Task")
    p[[2]] <- autoplot(outer_test_task) + ggtitle("Outer Test Task")
      # results of hyperparameter search
    p[[3]] <- autoplot(instance) + ggtitle("Hyperparameter Search")   # this is a patchwork, not a ggplot
        # performance on inner folds ROC
    #p[30] <- autoplot(rr_inner, type="roc")
        # perfomance on outer fold ROC
    p[[4]] <- autoplot(outer_prediction, type="roc") + ggtitle("Outer Test Performance")
    
    class(as.data.table(results[[1]][[1]][c("inner_performance","outer_performance")]) |> gt() |> fmt_number(decimals=3))
    
    # display the summary
    p[[1]] + p[[2]] + p[[4]] + plot_annotation( glrn$id )
    #p[[3]] + plot_annotation( paste(   
    
    #DEBUG
    stop("line #327")
    
  } #end loop over graph_learners 
  
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
    resampling = rsmp("cv", folds = NUM_OUTER_FOLDS),
    measure = measure,
    terminator = trm("evals", n_evals = 10) # change this
  )
  
  clin_rr <- resample(clin_task, clin_glm, rsmp("cv", folds = NUM_OUTER_FOLDS),
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

