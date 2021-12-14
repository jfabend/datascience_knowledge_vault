Permutation importance is a frequently used type of feature importance. It shows the drop in the score if the feature would be replaced with randomly permuted values. It is calculated with several straightforward steps:

1.  Train model with training data _X_train_, _y_train_;
2.  Make predictions for a training dataset _X_train_ — _y_hat_ and calculate the score — _score_ (higher score = better);
3.  To calculate permutation importance for each feature _feature_i_, do the following:  
    (1) permute _feature_i_ values in the training dataset while keeping all other features “as is” — _X_train_permuted_;  
    (2) make predictions using _X_train_permuted_ and previously trained model — _y_hat_permuted_;  
    (3) calculate the score on the permuted dataset — _score_permuted_;  
    (4) The importance of the feature is equal to _score_permuted — score_. Lower the delta — the more important the feature.
4.  The process is repeated several times to reduce the influence of random permutations and scores or ranks are averaged across runs.