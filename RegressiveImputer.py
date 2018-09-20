"""
RegressiveImputer.py
package regressive_imputer
version 1.0
created by AndrewJKing.com|@andrewsjourney

Pass in:
    A numpy array where some feature columns contain missing values,
    A second (test) array for which the same imputation should also be preformed
Processes:
    For each column with missing data:
        Find the rows where that column is not missing data
        Train a regression model on those rows
        Apply the regression model to the rows with the missing element to fill them in
Return:
    A numpy array where missing values have been imputed
    A second (test) where the same imputations have been performed
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_selection import f_classif
import warnings
import datetime
import pickle


def univariate_feature_selection(x_, y):
    """
    Returns list of columns that have a small univariate ANOVA p-value
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_val = f_classif(x_, y)
        keep_cols = np.where(p_val < 0.1)[0]
        if len(keep_cols) < 10:
            keep_cols = np.where(p_val < 0.2)[0]
        return keep_cols


def determine_if_binary(x):
    """
    Determines if passed array is binary or continuous
    """
    is_binary_column = False
    uniques = np.unique(x)
    if np.count_nonzero(np.isnan(uniques)) == uniques.shape[0] - 2:  # each nan is a unique value
        if uniques[0] == 0. and uniques[1] == 1.:
            is_binary_column = True
    return is_binary_column


def get_clean_columns(x):
    """
    This function returns the indcies of columns that are
     (1) not all null,
     (2) have more than one unique value,
     (3) are unique. 
    """
    # ## find columns that are not all nan
    not_all_nan_columns = np.where(~np.all(np.isnan(x), axis=0))[0]
    
    # ## find columns with only one unique value
    variant_columns = []
    for i in range(x.shape[1]):
        uniques = np.unique(x[:, i])
        if uniques.shape[0] > 1:
            variant_columns.append(i)

    # ## find the first occurence of each columns (eliminate duplicates)
    _, unique_columns = np.unique(x, axis=1, return_index=True)

    # ## keep items that are both not_all_nan_columns
    keep_columns = [v for v in variant_columns if v in not_all_nan_columns]

    keep_columns_2 = [v for v in keep_columns if v in unique_columns]

    return keep_columns_2


class ColumnTransform:
    """
    This class stores the transforms for an individual column
    """
    def __init__(self):
        self.column_transform = None
        self.column_transform_keep_col = []
        self.isSet = False

    def set_imputer(self, col_imputer):
        self.column_transform = col_imputer
        self.isSet = True

    def set_keep_column(self, keep_cols):
        self.column_transform_keep_col = keep_cols

    def return_keep_column(self):
        return self.column_transform_keep_col

    def predict(self, x):
        return self.column_transform.predict(x)

    def get_isSet(self):
        return self.isSet



class RegressiveImputer:
    """
    Class that holds the imputer
    """
    def __init__(self, col_transform_dir):
        self.number_columns_in = 0  # the number of columns passed into the transform (i.e., same as training set)
        self.number_columns_out = 0  # the number of columns returned after transform
        self.broad_imputer = Imputer(axis=0, missing_values='NaN', strategy='median', verbose=0)  # a braod imputer temporarily used across all the data
        self.broad_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)  # a broad scaller temporarily used across all the data
        self.column_transform_filenames = []  # the filenames for the indidual transform for each column. Each is stored to save memory
        self.column_transform_dir = col_transform_dir  # the location of the column transforms
        
    def fit(self, x_orig):
        """
        A x_broad is a imputed and scaled set of x_orig used to fit a column imputer
        """
        self.number_columns_in = x_orig.shape[1]
        x_broad_temp = self.broad_imputer.fit_transform(x_orig)  # fit broad imputer and transform
        x_broad = self.broad_scaler.fit_transform(x_broad_temp)  # fit broad scaler and transform

        # ## apply column wise data imputation ## #
        all_columns = [x for x in range(x_orig.shape[1])]
        progress_iterator = 0
        for curr_col in all_columns:
            # ## track progress
            if progress_iterator % 500 == 0:
                print (progress_iterator, '\t', datetime.datetime.now())
            progress_iterator += 1

            # ## create ColumnTransform
            curr_col_transform = ColumnTransform()
            filename_curr_col_transform = 'col_' + str(progress_iterator) + '.pkl'
            self.column_transform_filenames.append(filename_curr_col_transform)

            # ## find missing and not missing columns
            all_but_curr_col = all_columns[:curr_col] + all_columns[curr_col + 1:]
            x_orig_rows_not_missing = np.where(~np.isnan(x_orig[:, curr_col]))[0]
            x_orig_rows_with_missing = np.where(np.isnan(x_orig[:, curr_col]))[0]

            # ## sets need for training ## #
            x_broad_training_rows = x_broad[x_orig_rows_not_missing, :]  # reduce to rows w/ ! missing curr col
            x_broad_training_x = x_broad_training_rows[:, all_but_curr_col]  # remove curr_column for predictors
            x_orig_training_rows = x_orig[x_orig_rows_not_missing, :]
            x_orig_training_y = x_orig_training_rows[:, curr_col]  # rm all but curr_column for target

            # ## sets for imputation ## #
            x_broad_imputing_rows = x_broad[x_orig_rows_with_missing, :]  # reduce to rows with missing curr column
            x_broad_imputing_x = x_broad_imputing_rows[:, all_but_curr_col]  # remove all but curr_column for predictors

            # ## make sure there is at least one row with the missing curr_col filled in.
            if x_broad_training_x.shape[0] == 0:  # test for no training data
                # transform maintins None defaults
                print ('***Warning: some columns are all nan. Please run get_clean_columns on the input matrix before passing into imputer.***')
                continue

            # ## select informative features ## #
            keep_cols = univariate_feature_selection(x_broad_training_x, x_orig_training_y)
            if len(keep_cols):
                curr_col_transform.set_keep_column(keep_cols)
                x_broad_training_x = x_broad_training_x[:, keep_cols]
                x_broad_imputing_x = x_broad_imputing_x[:, keep_cols]
            else:
                # keep all columns
                curr_col_transform.set_keep_column(np.where(~np.all(np.isnan(x_broad_training_x), axis=0))[0])  ####point_2 3/4

            # determine if target attribute is binary
            if determine_if_binary(x_orig[:, curr_col]):
                col_imputer = LogisticRegression(penalty='l2', random_state=42)
            else:
                col_imputer = LinearRegression()
            
            # fit model, store it, apply it if curr_col has missing
            col_imputer.fit(x_broad_training_x, x_orig_training_y)
            curr_col_transform.set_imputer(col_imputer)  ####point 2/4
            if x_orig_rows_with_missing.size:
                x_orig[x_orig_rows_with_missing, curr_col] = col_imputer.predict(x_broad_imputing_x)  # impute

            # ## store column transform
            pickle.dump(curr_col_transform, open(self.column_transform_dir + filename_curr_col_transform, 'wb')) 

        self.keep_columns = get_clean_columns(x_orig)
        self.number_columns_out = len(self.keep_columns)
        return

    def transform(self, x_orig):
        if self.number_columns_in != x_orig.shape[1]:
            print ("***Warning - column dimentions do not match. Returning original.***")
            return x_orig
        
        x_broad_temp = self.broad_imputer.transform(x_orig)  # fit broad imputer and transform
        x_broad = self.broad_scaler.transform(x_broad_temp)  # fit broad scaler and transform

        # ## apply column wise data imputation ## #
        all_columns = [x for x in range(x_orig.shape[1])]
        for col_idx, curr_col in enumerate(all_columns):
            # load column transform 
            curr_col_transform = pickle.load(open(self.column_transform_dir + self.column_transform_filenames[col_idx], 'rb'))
            
            if not curr_col_transform.get_isSet():  # see if transform is avaliable    ####point 3/4
                continue   # not transform avaliable for this column (was always missing in trainig data)

            all_but_curr_col = all_columns[:curr_col] + all_columns[curr_col + 1:]
            x_orig_rows_with_missing = np.where(np.isnan(x_orig[:, curr_col]))[0]

            if not len(x_orig_rows_with_missing):  # see if column is missing any data
                continue  # column does not have any rows with missing data

            # ## sets for imputation and apply imputation ## #
            x_broad_imputing_rows = x_broad[x_orig_rows_with_missing, :]  # reduce to rows with missing curr column
            x_broad_imputing_x = x_broad_imputing_rows[:, curr_col_transform.return_keep_column()]  # reduce predictors to trained on columns   
            x_orig[x_orig_rows_with_missing, curr_col] = curr_col_transform.predict(x_broad_imputing_x)  # impute  

        return x_orig[:, self.keep_columns]

    def fit_transform(self, x_orig):
        self.fit(x_orig)
        return self.transform(x_orig)

    def transform_column_names(self, names):
        if self.number_columns_in != len(names):
            print ("***Warning - column dimentions do not match. Returning original names***")
            return names
        return names[self.keep_columns]
