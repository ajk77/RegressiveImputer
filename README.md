# regressive-imputer
Impute missing values via a regression model


Run this imputer on any numpy two dimensional array. Samples must be rowwise.<br>
It is recommended that you run get_clean_columns() on your data before passing it to RegressiveImputer. If you data contains any columns that are all nan and you do not run get_clean_columns, then the columns will remain nan in the transformed dataset.