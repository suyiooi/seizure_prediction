Running on Benji notes:
the mlr3 version is 0.21.0, R version 4.4.1

Parallelisation:
future::plan("multisession") runs into errors with 'col_roles' so I changed this to future:plan("sequential") which works

Encapsulation and fallback:
in mlr3 version 0.20.0: learner$encapsulate(method = "evaluate", fallback = lrn("classif.log_reg", predict_type = "prob"))
in mlr3 version 0.21.0, they are coded differently: learner$encapsulate = c(train = "evaluate", predict = "evaluate")
learner$fallback = lrn("classif.log_reg", predict_type = "prob")
Why is it defaulting to the fallback in 0.21.0
