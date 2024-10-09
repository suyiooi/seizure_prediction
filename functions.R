#functions for nested_CV
#Author: Suyi Ooi
#Date: 19/09/2024

#extract clinical factors
extract_clin_factors <- function(dat){
  clin <- subset(dat, select = c(szrec1,sex,age,mri,
                                 nocturnal_sz,
                                 risk_factors___2,risk_factors___3,
                                 risk_factors___9,sz_type,eeg,BrainPAD))
  return(clin)
}

#group features for combat
extract_featuregroup_list <- function( dat ) {
  #make lists of train and test df
  df_list <- list()
  
  feature_groups <- list("^meanthickness",
                         "vol_[lr]_putamen|vol_[lr]_pallidum|_cc_",
                         "^subcort(?!gm)","^foldingindex_",
                         "^[^f].*_foldingindex$", "meangaus",
                         "^gmvol", "vol_.?_gm", "totalgm_vol", 
                         "^[lr]_total_sa$", "^lgi_mean.?", "ICI", "pct", ".?_cortex_vol", 
                         "cerebralwm_vol", "^[lr]_cortex_wm_sa", "icv", "subcortgm_vol", 
                         "h_meancurv", "k1_meancurv", "k2_meancurv", "SI_meancurv", 
                         "air_gaussian", "intrincurv", "k1_rect_surf_integral", 
                         "k1_nat_surf_integral", "k2_rect_surf_integral", 
                         "k2_nat_surf_integral", "FI_nat_surf_integral", 
                         "FI_rect_surf_integral", "^asym")
  
  for (var in feature_groups){
    dat_vars_train = as.data.frame(dat[ , grepl(var, names(dat), perl = TRUE)])
    if (ncol(dat_vars_train) == 1) {
      colnames(dat_vars_train) <- var }
    df_list[[var]] <- dat_vars_train
  }
  
  return( df_list )     
}

#combat train
do_nc_featuregroup <- function(df, nc_batch, nc_mod) {
  data_transposed <- t(df)
  
  if (nrow(data_transposed) == 1)
    use_eb <- FALSE
  else
    use_eb <- TRUE
  
  # Apply neuroCombat
  combat_output <- neuroCombat(dat = data_transposed, 
                               batch = nc_batch,  
                               mod = nc_mod, 
                               eb = use_eb)
  
  return( list( dat.combat = as.data.frame(combat_output$dat.combat),
                estimates = combat_output$estimates))
}

do_neuroCombat <- function(train_df_list, batch, mod ){
  
  combat <- lapply(train_df_list, do_nc_featuregroup, batch, mod )
  return(combat)
  
}

#combat test
do_nc_featuregroup_FromTraining <- function(test.df, test.batch, train.estimates) {
  
  test_data_transposed <- t(test.df)
  this_featuregroup <- substring(names(test.df)[1],1,4)  # get the string "varX"
  
  combat_test_output <- neuroCombatFromTraining(dat = test_data_transposed, 
                                                batch = test.batch,
                                                estimates = train.estimates, 
                                                mod = NULL)
  return( combat_test_output )
}

do_neuroCombatFromTraining <- function(test_df_list, test.batch, train.estimates ) {
  output_list <- list()
  for (ind in 1:length(test_df_list)) {
    combat.test <-  do_nc_featuregroup_FromTraining( test_df_list[[ind]], 
                                                     test.batch, 
                                                     train.estimates[[ind]] )
    output_list <- append(output_list, list(as.data.frame(combat.test$dat.combat)) )
    
  }
  return(output_list)
}

#turn the combatted list into dataframe
get_dat_combated_df <- function(dat.combat.df.train.test.list){
  transposed_cb.dat <- list()
  
  for (i in seq_along(dat.combat.df.train.test.list)) {
    transposed_cb.dat[[i]] <- as.data.frame(t(dat.combat.df.train.test.list[[i]]))
  }
  
  dat.combated <- bind_cols(transposed_cb.dat)
  return(as.data.frame(dat.combated))
}

# Create stratified folds with balanced y outcome
stratified_folds <- function(task, k) {
  target <- task$truth()
  folds <- vector("list", k)
  for (class in unique(target)) {
    class_indices <- which(target == class)
    class_indices <- sample(class_indices)
    fold_indices <- split(class_indices, cut(seq_along(class_indices), k, labels = FALSE))
    for (i in seq_len(k)) {
      folds[[i]] <- c(folds[[i]], fold_indices[[i]])
    }
  }
  folds
}

#extract inner and outer AUC (mean)
extract_auc <- function(results, loop){
  
  if (as.character(loop) == "outer"){
    
    outer_performances <- as.data.frame(sapply(1:10, function(i) {
      sapply(1:12, function(j) {
        results[[i]][[j]][[1]][["outer_performance"]]
      })
    }))
    
    mean_auc <- as.numeric(c())
    
    for (k in nrow(outer_performances)){
      mean_auc <- sapply(1:12, function(k) {mean(as.numeric(outer_performances[k,1:10]))
      })
    }
    
    outer_performances$mean_auc <- mean_auc
    colnames(outer_performances) <- c(paste0("Fold", 1:10),
                                      paste0("mean_outer_auc"))
    rownames(outer_performances) <- sapply(1:12, function(j) {
      results[[i]][[j]][[1]][["learner"]]
    })
    
    return(outer_performances)
    
  } else {
    inner_performances <- as.data.frame(sapply(1:10, function(i) {
      sapply(1:12, function(j){
        
        results[[i]][[j]][[1]][["inner_performance"]]
      })
    }))
    
    mean_auc <- as.numeric(c())
    
    for (k in nrow(inner_performances)){
      mean_auc <- sapply(1:12, function(k) { mean(as.numeric(inner_performances[k,1:10]))
      })
    }
    
    inner_performances$mean_auc <- mean_auc
    colnames(inner_performances) <- c(paste0("Fold", 1:10),
                                      paste0("mean_inner_auc"))
    rownames(inner_performances) <- sapply(1:12, function(j) {
      results[[i]][[j]][[1]][["learner"]]
    })
  }
  return(inner_performances)
}

#top 4 models + clinical AUC (stacked)
stack_top4_predictions <- function(results, models) {
  
  pred_stacks <- list(
    pred_stack_rf = list(),
    pred_stack_xgb = list(),
    pred_stack_glm = list(),
    pred_stack_svm = list(),
    pred_stack_clin = list()
  )
  
  rocs <- list()
  
  for (i in seq_len(10)) {  
    for (j in seq_len(12)){
      learner <- results[[i]][[j]][[1]][["learner"]]
      
      if (learner %in% names(models)) {
        model_list <- models[[learner]]
        
        pred_stacks[[model_list]] <- c(
          pred_stacks[[model_list]], 
          list(results[[i]][[j]][[1]][["outer_prediction"]])
        )
      }
    }
    
    #clinical model
    pred_stacks$pred_stack_clin <- c(pred_stacks$pred_stack_clin, clin_pred[[i]][[1]])
  }
  
  rf_prob <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_rf, function(x) as.data.frame(x$prob))))
  rf_truth <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_rf, function(x) as.data.frame(x$truth))))
  
  svm_prob <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_svm, function(x) as.data.frame(x$prob))))
  svm_truth <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_svm, function(x) as.data.frame(x$truth))))
  
  xgb_prob <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_xgb, function(x) as.data.frame(x$prob))))
  xgb_truth <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_xgb, function(x) as.data.frame(x$truth))))
  
  glm_prob <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_glm, function(x) as.data.frame(x$prob))))
  glm_truth <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_glm, function(x) as.data.frame(x$truth))))
  
  clin_prob <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_clin, function(x) as.data.frame(x$prob))))
  clin_truth <- as.data.frame(do.call(rbind, lapply(pred_stacks$pred_stack_clin, function(x) as.data.frame(x$truth))))
  
  #roc list
  rocs <- c(rocs, list(
    rf_stacked_roc = roc(rf_prob$yes, response = rf_truth$`x$truth`, plot = TRUE, print.auc = TRUE, ci = TRUE),
    xgb_stacked_roc = roc(xgb_prob$yes, response = xgb_truth$`x$truth`, plot = TRUE, print.auc = TRUE, ci = TRUE),
    glm_stacked_roc = roc(glm_prob$yes, response = glm_truth$`x$truth`, plot = TRUE, print.auc = TRUE, ci = TRUE),
    svm_stacked_roc = roc(svm_prob$yes, response = svm_truth$`x$truth`, plot = TRUE, print.auc = TRUE, ci = TRUE),
    clin_stacked_roc = roc(clin_prob$yes, response = clin_truth$`x$truth`, plot = TRUE, print.auc = TRUE, ci = TRUE)
  ))
  
  return(rocs)
}

#plot rocs
plot_rocs <- function(predictions){
  
  roc_plot <- ggplot()
  all_roc_data <- data.frame()
  
  for (roc in seq_along(predictions)) {
    roc_curve <- ggroc(predictions[[roc]])
    roc_data <- roc_curve$data
    roc_data$model <- labels[roc]  
    all_roc_data <- rbind(all_roc_data, roc_data) 
  }
  
  roc_plot <- ggplot(all_roc_data, aes(x = 1 - specificity, y = sensitivity, color = model)) +
    geom_line(size = 1) + 
    labs(title = "AUC-ROC", 
         x = "1 - Specificity", 
         y = "Sensitivity") +
    scale_color_manual(name = NULL, values = colors, labels = labels) + 
    theme_minimal() +
    theme(axis.text.x = element_text(colour = "black", size = 14),
          axis.text.y = element_text(colour = "black", size = 14),
          axis.title.x = element_text(size = 13),
          axis.title.y = element_text(size = 13),
          legend.position = c(0.61, 0.15), 
          legend.text = element_text(size = 11))
  
  return(roc_plot)
}

#extract_best_param per algorithm
extract_best_param <- function(results, model) {
  
  if (as.character(model) == "random_forest") {
    
    rf_best_param <- sapply(1:10, function(i) {
      
      sapply(1:3, function(j) {
        
        rf_param <- results[[i]][[j]][[1]][["best_params"]]
        
        rf_grep <- rf_param[grep(".filter.frac|num.trees|mtry", names(rf_param))]
        
        if (length(grep("filter.frac", names(rf_param))) == 0) {
          
          rf_param_subset <- rf_param[grep("num.trees|mtry", names(rf_param))]
          rf_param_subset[".filter.frac"] <- NA
          
        } else {
          rf_param_subset <- rf_param[grep(".filter.frac|num.trees|mtry", names(rf_param))]
        }
        
        return(rf_param_subset)
      })
      
    })
    return(rf_best_param)
    
  } else if (as.character(model) == "xgboost") {
    
    xgb_best_param <- as.data.frame(sapply(1:10, function(i) {
      
      sapply(4:6, function(j) {
        
        xgb_param <- results[[i]][[j]][[1]][["best_params"]]
        
        xgb_grep <- xgb_param[grep(".filter.frac|nrounds|eta|max_depth|min_child_weight|gamma|subsample|colsample_bytree", names(xgb_param))]
        
        if (length(grep("filter.frac", names(xgb_param))) == 0) {
          
          xgb_param_subset <- xgb_param[grep("nrounds|eta|max_depth|min_child_weight|gamma|subsample|colsample_bytree", names(xgb_param))]
          xgb_param_subset[".filter.frac"] <- NA
          
        } else {
          
          xgb_param_subset <- xgb_grep
        }
        
        return(xgb_param_subset)
      })
      
    }))
    return(xgb_best_param)
    
  } else if (as.character(model) == "lasso") {
    
    lasso_best_param <- as.data.frame(sapply(1:10, function(i) {
      
      sapply(7:9, function(j) {
        
        lasso_param <- results[[i]][[j]][[1]][["best_params"]]
        
        lasso_grep <- lasso_param[grep(".filter.frac|lambda", names(lasso_param))]
        
        if (length(grep(".filter.frac", names(lasso_param))) == 0) {
          
          lasso_param_subset <- lasso_param[grep("lambda", names(lasso_param))]
          lasso_param_subset[".filter.frac"] <- NA
          
        } else {
          
          lasso_param_subset <- lasso_grep
        }
        
        return(lasso_param_subset)
      })
    }))
    return(lasso_best_param)
    
  } else if (as.character(model) == "svm") {
    
    svm_best_param <- as.data.frame(sapply(1:10, function(i) {
      
      sapply(10:12, function(j) {
        
        svm_param <- results[[i]][[j]][[1]][["best_params"]]
        
        svm_grep <- svm_param[grep(".filter.frac|.cost", names(svm_param))]
        
        if (length(grep(".filter.frac", names(svm_param))) == 0) {
          
          svm_param_subset <- svm_param[grep(".cost", names(svm_param))]
          
          svm_param_subset[".filter.frac"] <- NA
          
        } else {
          
          svm_param_subset <- svm_grep
        }
        
        return(svm_param_subset)
      })
    }))
    return(svm_best_param)
  }
}


    
    #---hyperparameters---#


create_param_sets <- function(glrn_id) {
  
  glrn_id = glrn$id
  
  #perm_randomforest  
  if (grepl("perm.classif.ranger", glrn_id)) {
    param_set <- ps(
      perm.classif.ranger.classif.ranger.num.trees = p_int(50,250),
      
      perm.classif.ranger.classif.ranger.mtry = p_int(10,25),
      perm.classif.ranger.permutation.filter.frac = p_dbl(0.05,0.3),
      perm.classif.ranger.classif.ranger.num.threads = p_int(4,4) #benji has 32 CPU and 2 threads per core
    )
    
    #pca_randomforest  
  } else if (grepl("pca.classif.ranger", glrn_id)) {
    param_set <- ps(
      pca.classif.ranger.classif.ranger.num.trees = p_int(50,250),
      pca.classif.ranger.classif.ranger.mtry = p_int(10,25),
      pca.classif.ranger.variance.filter.frac = p_dbl(0.5,0.9),
      pca.classif.ranger.classif.ranger.num.threads = p_int(4,4)
    )
    
    #auc_randomforest   
  } else if (grepl("auc_filter.classif.ranger", glrn_id)) {
    param_set <- ps(
      auc_filter.classif.ranger.classif.ranger.num.trees = p_int(50,250),
      auc_filter.classif.ranger.classif.ranger.mtry = p_int(10,25),
      auc_filter.classif.ranger.auc.filter.frac = p_dbl(0.1,0.3),
      auc_filter.classif.ranger.classif.ranger.num.threads = p_int(4,4)
    )
    
    #nofilter_randomforest   
  } else if (grepl("no_filter.classif.ranger", glrn_id)) {
    param_set <- ps(no_filter.classif.ranger.classif.ranger.num.trees = p_int(50,250),
                    no_filter.classif.ranger.classif.ranger.mtry = p_int(10,25),
                    no_filter.classif.ranger.classif.ranger.num.threads = p_int(4,4)
    )
    
    #perm_xgb    
  #} else if (grepl("perm.classif.xgboost", glrn_id)) {
    
    #param_set <- ps(
      #perm.classif.xgboost.classif.xgboost.nrounds = p_int(50,250),
      #perm.classif.xgboost.classif.xgboost.eta = p_dbl(1e-4, 0.4),
      #perm.classif.xgboost.classif.xgboost.max_depth = p_int(1,15),
      #perm.classif.xgboost.classif.xgboost.min_child_weight = p_int(1,10),
      #perm.classif.xgboost.classif.xgboost.gamma = p_dbl(1e-4, 10),
      #perm.classif.xgboost.classif.xgboost.alpha = p_int(1,1),
      #perm.classif.xgboost.classif.xgboost.subsample = p_dbl(0.5,0.8),
      #perm.classif.xgboost.classif.xgboost.colsample_bytree = p_dbl(0.5,0.8),
      #perm.classif.xgboost.permutation.filter.frac = p_dbl(0.05,0.3),
      #perm.classif.xgboost.classif.xgboost.nthread = p_int(4,4)
    #)
    
    #pca_xgb    
  } else if (grepl("pca.classif.xgboost", glrn_id)) {
    param_set <- ps(
      pca.classif.xgboost.classif.xgboost.nrounds = p_int(50,250),
      pca.classif.xgboost.classif.xgboost.eta = p_dbl(1e-4, 0.4),
      pca.classif.xgboost.classif.xgboost.max_depth = p_int(1,15),
      pca.classif.xgboost.classif.xgboost.min_child_weight = p_int(1,10),
      pca.classif.xgboost.classif.xgboost.gamma = p_dbl(1e-4, 10),
      pca.classif.xgboost.classif.xgboost.alpha = p_int(1,1),
      pca.classif.xgboost.classif.xgboost.subsample = p_dbl(0.5,0.8),
      pca.classif.xgboost.classif.xgboost.colsample_bytree = p_dbl(0.5, 0.8),
      pca.classif.xgboost.variance.filter.frac = p_dbl(0.5, 0.9),
      pca.classif.xgboost.classif.xgboost.nthread = p_int(4,4)
    )
    
    #auc_filter_xgb    
  } else if (grepl("auc_filter.classif.xgboost", glrn_id)) {
    
    param_set <- ps(
      auc_filter.classif.xgboost.classif.xgboost.nrounds = p_int(50,250),
      auc_filter.classif.xgboost.classif.xgboost.eta = p_dbl(1e-4, 0.4),
      auc_filter.classif.xgboost.classif.xgboost.max_depth = p_int(1,15),
      auc_filter.classif.xgboost.classif.xgboost.min_child_weight = p_int(1,10),
      auc_filter.classif.xgboost.classif.xgboost.gamma = p_dbl(1e-4, 10),
      auc_filter.classif.xgboost.classif.xgboost.alpha = p_int(1,1),
      auc_filter.classif.xgboost.classif.xgboost.subsample = p_dbl(0.5,0.8),
      auc_filter.classif.xgboost.classif.xgboost.colsample_bytree = p_dbl(0.5, 0.8),
      auc_filter.classif.xgboost.auc.filter.frac = p_dbl(0.1, 0.3),
      auc_filter.classif.xgboost.classif.xgboost.nthread = p_int(4,4)
    )
    
    #nofilter_xgb    
  } else if (grepl("no_filter.classif.xgboost", glrn_id)) {
    
    param_set <- ps(
      no_filter.classif.xgboost.classif.xgboost.nrounds = p_int(50,250),
      no_filter.classif.xgboost.classif.xgboost.eta = p_dbl(1e-4, 0.4),
      no_filter.classif.xgboost.classif.xgboost.max_depth = p_int(1,15),
      no_filter.classif.xgboost.classif.xgboost.min_child_weight = p_int(1,10),
      no_filter.classif.xgboost.classif.xgboost.gamma = p_dbl(1e-4, 10),
      no_filter.classif.xgboost.classif.xgboost.alpha = p_int(1,1),
      no_filter.classif.xgboost.classif.xgboost.subsample = p_dbl(0.5,0.8),
      no_filter.classif.xgboost.classif.xgboost.colsample_bytree = p_dbl(0.5, 0.8),
      no_filter.classif.xgboost.classif.xgboost.nthread = p_int(4,4)
    )
    
    #perm_glm
  #} else if (grepl("perm.classif.glmnet", glrn_id)) {
    
    #param_set <- ps(
      #perm.classif.glmnet.classif.glmnet.lambda = p_dbl(0.001,0.20),
      #perm.classif.glmnet.permutation.filter.frac = p_dbl(0.03,0.5)
   # )
    
    #pca_glm     
  } else if (grepl("pca.classif.glmnet", glrn_id)) {
    
    param_set <- ps(
      pca.classif.glmnet.classif.glmnet.lambda = p_dbl(0.001,0.20),
      pca.classif.glmnet.variance.filter.frac = p_dbl(0.5,0.9)
    )
    
    #auc_filter_glm     
  } else if (grepl("auc_filter.classif.glmnet", glrn_id)) {
    
    param_set <- ps(
      auc_filter.classif.glmnet.classif.glmnet.lambda = p_dbl(0.001,0.20),
      auc_filter.classif.glmnet.auc.filter.frac = p_dbl(0.1,0.3)
    )
    
    #no_filter_glm    
  } else if (grepl("no_filter.classif.glmnet", glrn_id)) {
    
    param_set <- ps(
      no_filter.classif.glmnet.classif.glmnet.lambda = p_dbl(0.001, 0.20)
    )
    
    # perm_svm
  #} else if (grepl("perm.classif.svm", glrn_id)) {
    
    #param_set <- ps(
      #perm.classif.svm.classif.svm.cost = p_dbl(2^-5, 2^5),
      #perm.classif.svm.permutation.filter.frac = p_dbl(0.03, 0.5)
    #)
    
    #pca_svm    
  } else if (grepl("pca.classif.svm", glrn_id)) {
    
    param_set <- ps(
      pca.classif.svm.classif.svm.cost = p_dbl(2^-5, 2^5),
      pca.classif.svm.variance.filter.frac = p_dbl(0.5, 0.9)
    )
    
    #auc_filter_svm    
  } else if (grepl("auc_filter.classif.svm", glrn_id)) {
    
    param_set <- ps(
      auc_filter.classif.svm.classif.svm.cost = p_dbl(2^-5, 2^5),
      auc_filter.classif.svm.auc.filter.frac = p_dbl(0.1, 0.3)
    )
    
    #no_filter_svm    
  } else if (grepl("no_filter.classif.svm", glrn_id)) {
    
    param_set <- ps(
      no_filter.classif.svm.classif.svm.cost = p_dbl(2^-5, 2^5)
    )
    
  } 
}  

