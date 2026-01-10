Maximize Recall at FP ≤ 0.1%

Baseline LogisticRegression

Precision: 0.0587, Recall: 0.9061, F1-Score: 0.1102
CM:
[[221722   5729]
 [    37    357]]



--------------------------------------------
LogReg with chosen threshold

chosen threshold 
Chosen threshold: 0.9898731402482259 using strategy: Max recall with FPR≤0.0010
Metrics at chosen threshold:
threshold: 0.9899
fpr: 0.0010
recall: 0.8325
precision: 0.5942
f1: 0.6934

Chosen threshold cm:
Confusion Matrix:
[[227227    224]
 [    66    328]]

HGB with chosen threshold

Chosen threshold: 0.051411170528817586 using strategy: Max recall with FPR≤0.0010
Metrics at chosen threshold:
threshold: 0.0514
fpr: 0.0010
recall: 0.8299
precision: 0.5935
f1: 0.6921

----------------------------------------
HGB tuning + sample weight=200 best parameters and score 

Best Parameters: {'classifier__learning_rate': 0.19262268462637636, 'classifier__max_iter': 312, 'classifier__max_leaf_nodes': 30, 'classifier__min_samples_leaf': 17}
Best Score (Recall at FPR 0.1%): 0.7131450827653358

Chosen threshold after hyperparameter tuning: 0.5904002134695905, strategy: Max recall with FPR≤0.0010
Metrics at chosen threshold after hyperparameter tuning:
threshold: 0.5904
fpr: 0.001
recall: 0.8376
precision: 0.6022
f1: 0.7006

------------------------------
HGB + tuning + calibration 

Chosen threshold after hyperparameter tuning: 0.009550051827792954, strategy: Max recall with FPR≤0.0010
Metrics at chosen threshold after hyperparameter tuning:
threshold: 0.0096
fpr: 0.001
recall: 0.8223
precision: 0.5912
f1: 0.6879



