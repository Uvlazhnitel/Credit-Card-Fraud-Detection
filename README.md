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

---------------------------
HGB baseline 

Chosen threshold: 0.603880655997441, strategy: Max recall with FPR≤0.0010
Metrics at chosen threshold:
threshold: 0.6039
fpr: 0.0009
recall: 0.7005
precision: 0.586
f1: 0.6382

Confusion Matrix:
 [[227256    195]
 [   118    276]]


------------------------------------
HGB + randomsearch 

Chosen threshold after hyperparameter tuning: 0.011140445817131722, strategy: Max recall with FPR≤0.0010
Metrics at chosen threshold after hyperparameter tuning:
threshold: 0.0111
fpr: 0.0009
recall: 0.8223
precision: 0.6231
f1: 0.709
Confusion Matrix after hyperparameter tuning:
 [[227255    196]
 [    70    324]]

------------------------------
HGB + tuning + calibration 

Chosen threshold after hyperparameter tuning: 0.4502900283651993, strategy: Max recall with FPR≤0.0010
Metrics at chosen threshold after hyperparameter tuning:
threshold: 0.4503
fpr: 0.0009
recall: 0.8452
precision: 0.6077
f1: 0.707
Best Parameters: {'classifier__learning_rate': np.float64(0.18850479889719593), 'classifier__max_iter': 545, 'classifier__max_leaf_nodes': 28, 'classifier__min_samples_leaf': 20}


