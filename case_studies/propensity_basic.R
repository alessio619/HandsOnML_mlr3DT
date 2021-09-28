



# : =================================================================================


seed(619)

# 0. PACKAGES ============================================

library(mlr3verse)
library(magrittr)
library(here)
library(data.table)


# A. DATA UPLOAD =========================================

dtw = readRDS(file = here('data', 'data_welfare.rds'))



### Check columns dna
skimr::skim(dtw)




# : =================================================================================




# B. mlr3 in Action  ===================================

## 01. Task Creation -------------------

tsk_hea = as_task_classif(dtw, target = 'welf_healthcare', id = 'healthcare')
#tsk_ass = as_task_classif(dtw, target = 'welf_assistance', id = 'assitance')
#tsk_oth = as_task_classif(dtw, target = 'welf_other', id = 'other')

autoplot(tsk_hea, type = 'pairs')

### Task API ------------------------

# Shape
tsk_hea$nrow
tsk_hea$ncol

### Retrieve data
tsk_hea$data() %>% head()
tsk_hea$data(rows = 1:6) 

### Meta Data
tsk_hea$feature_names
tsk_hea$target_names

print(tsk_hea$col_roles)

### to modify
tsk_hea$col_roles$stratum = 'workers'
print(tsk_hea$col_roles)

tsk_hea$set_col_roles('geo', roles = 'stratum')
print(tsk_hea$col_roles)


# Task Mutators
tsk_hea$cbind(data.frame(ids = 1:5747))
tsk_hea$head()

tsk_hea$rbind(tsk_hea$head())
tsk_hea$nrow

tsk_hea$select(c('cod_nace', 'income', 'workers'))
tsk_hea$filter(1:5747)




## 02. Learner -------------------

lrn.log_reg = lrn('classif.log_reg')



## 03A. Train-Validate-Test -------------------

train_set = sample(tsk_hea$nrow, 0.8 * tsk_hea$nrow)
test_set = setdiff(seq_len(tsk_hea$nrow), train_set)


## 04A. Train and Predict the Learner -----------------------------

lrn.log_reg$train(tsk_hea, row_ids = train_set) # predict_type = 'prob'
print(lrn.log_reg$model)

prd.log_reg = lrn.log_reg$predict(tsk_hea, row_ids = test_set)
print(prd.log_reg)

head(as.data.table(prd.log_reg))

# Observe Predictions 

### Confusion Matrix
prd.log_reg$confusion

### Plot
autoplot(prd.log_reg)


## 05A. Performance Measurement -----------------------------

### Select measurements
prfm = msrs(c('classif.acc', 'classif.recall', 'classif.precision', 'classif.sensitivity', 'classif.specificity', 'classif.ce'))

prd.log_reg$score(prfm)




# : ===============================================================================




## 03B. Train-Validate-Test -------------------

rsp = rsmp('bootstrap')
rsp$param_set$values = list(ratio = 0.75, repeats = 20)

print(rsp)

rsp$instantiate(tsk_hea)

# check iters
rsp$iters


## 04B. Train and Predict the Learner -----------------------------

rr.log_leg = resample(tsk_hea, lrn.log_reg, rsp, store_models = TRUE)

# Observe Predictions 
rr.log_leg$aggregate(prfm)

rr.log_leg$score(prfm)


### Plot
autoplot(rr.log_leg$prediction())



## 05B. Retrieve the Learner --------------------------------------

lrnR.log_reg = rr.log_leg$learners[[1]]
lrnR.log_reg$model





# : ===============================================================================




# C. BENCHAMARKING ================================================================

## 01. Desing Creation --------------------------------------

### Learners to Benchmark
lrn.nv = lrn('classif.naive_bayes')
lrn.kknn = lrn('classif.kknn')
lrn.lda = lrn('classif.lda')
lrn.qda = lrn('classif.qda')
lrn.ranger = lrn('classif.ranger')




### The Benchmark  lrn.cv_glmnet
dsg = benchmark_grid(tasks = tsk_hea,
                     learners = c(lrn.log_reg, lrn.kknn, lrn.lda, lrn.ranger, lrn.qda, lrn.nv), 
                     resamplings = rsmps('cv', folds = 5))

print(dsg)


## 02. Execute Benchmark -----------------------------------

bmr.hea = benchmark(dsg)

bmr.hea$aggregate(prfm)

autoplot(bmr.hea)




### FOR ROC

lrn.nv = lrn('classif.naive_bayes', predict_type = 'prob')
lrn.kknn = lrn('classif.kknn', predict_type = 'prob')
lrn.lda = lrn('classif.lda', predict_type = 'prob')
lrn.qda = lrn('classif.qda', predict_type = 'prob')
lrn.ranger = lrn('classif.ranger', predict_type = 'prob')

dsg.2 = benchmark_grid(tasks = tsk_hea,
                     learners = c(lrn.log_reg, lrn.kknn, lrn.lda, lrn.ranger, lrn.nv), 
                     resamplings = rsmps('repeated_cv'))

bmr.2 = benchmark(dsg.2)

bmr.2$aggregate(prfm)
autoplot(bmr.2)


autoplot(bmr.2, type = 'roc')




# D. Tuning ================================================================

### Observe params to tune
lrn.lda$param_set
lrn.kknn$param_set

### Create the bounded space 
sp_kknn = ps(
    k = p_int(lower = 1, upper = 14),
    kernel = p_fct(levels = c('rectangular', 'optimal', 'gaussian'))
)

hout = rsmp('holdout')
measure = msr('classif.acc')
evals30 = trm('evals', n_evals = 40)

instance = TuningInstanceSingleCrit$new(
    task = tsk_hea,
    learner = lrn.kknn,
    resampling = hout,
    measure = measure,
    search_space = sp_kknn,
    terminator = evals30
)


tuner = tnr('grid_search', resolution = 10)

tuner$optimize(instance)


instance$result_learner_param_vals
instance$result_y


instance$archive$benchmark_result$score(msr('classif.tpr'))
