"""
ImputeDriver.py
package RegressiveImputer
version 1.1
created by AndrewJKing.com|@andrewsjourney

This program demonstrates how to run RegressiveImputer.

It also demonstrates code for splitting the data into five folds.  

To run, you will need to replace '/my_base_dir/' in __main__.
"""

from RegressiveImputer import RegressiveImputer, get_clean_columns
from sklearn.preprocessing import Imputer
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pickle


def load_list(file_name):
    loaded_list = []
    with open(file_name, 'r') as f:
        for full_line in f:
            line = full_line.rstrip()
            if line:
                loaded_list.append(line)
    return loaded_list


def imputation(params):
    """
    This function runs feature column cleaning and imputation
    """

    # ## load data
    full_training_data = np.load(params['assemble_output_filename'] + '.npy')
    full_training_names = np.load(params['assemble_output_filename'] + '_names.npy')
    full_testing_data = np.load(params['assemble_eval_output_filename'] + '.npy')
    full_testing_names = np.load(params['assemble_eval_output_filename'] + '_names.npy')
    print ('Train data loaded: ', full_training_data.shape, full_training_names.shape)
    print ('Eval data loaded: ', full_testing_data.shape, full_testing_names.shape)
    print ('='*20)

    # ## clean columns to eliminate all nan, only one not nan, and non-uniques
    keep_columns = get_clean_columns(full_training_data)
    cleaned_training_data = full_training_data[:, keep_columns]
    cleaned_training_names = full_training_names[keep_columns, :]
    cleaned_testing_data = full_testing_data[:, keep_columns]
    cleaned_testing_names = full_testing_names[keep_columns, :]
    pickle.dump(keep_columns, open(params['keep_columns_out'], 'wb'))
    print ('Train columns cleaned: ', cleaned_training_data.shape, cleaned_training_names.shape)
    print ('Eval columns cleaned: ', cleaned_testing_data.shape, cleaned_testing_names.shape)
    print ('='*20)

    if params['unit_testing']:
        '''
        This reduces the feature columns for faster runtime and visual inspection
        '''
        keep_columns = [0,1,2,3,4,5,6,7,999,9000]  # enter any desired columns here
        cleaned_training_data = cleaned_training_data[:, keep_columns]
        cleaned_training_names = cleaned_training_names[keep_columns, :]
        cleaned_testing_data = cleaned_testing_data[:, keep_columns]
        cleaned_testing_names = cleaned_testing_names[keep_columns, :]
        print ('tests: ', cleaned_training_data.shape, cleaned_training_names.shape)
        print (cleaned_training_data[1:5,8], cleaned_testing_data[1:5,8])
        print ('='*20)

    # ## use regressive imputer
    r_imputer = RegressiveImputer(params['r_imputer_out'] + '/')
    r_imputed_training_data = r_imputer.fit_transform(np.copy(cleaned_training_data))
    r_imputed_testing_data = r_imputer.transform(np.copy(cleaned_testing_data))
    r_imputed_training_names = r_imputer.transform_column_names(np.copy(cleaned_training_names))
    r_imputed_test_names = r_imputer.transform_column_names(np.copy(cleaned_testing_names))
    pickle.dump(r_imputer, open(params['r_imputer_out'] + '.pkl', 'wb'))
    np.save(params['assemble_output_filename'] + '_rImp', r_imputed_training_data)
    np.save(params['assemble_output_filename'] + '_rImp_names', r_imputed_training_names)
    np.save(params['assemble_eval_output_filename'] + '_rImp', r_imputed_testing_data)
    np.save(params['assemble_eval_output_filename'] + '_rImp_names', r_imputed_test_names)

    print ('Train rImp: ', r_imputed_training_data.shape, r_imputed_training_names.shape, r_imputed_training_data.shape[1] == r_imputed_training_names.shape[0])
    print ('Eval rImp: ', r_imputed_testing_data.shape, r_imputed_test_names.shape, r_imputed_testing_data.shape[1] == r_imputed_test_names.shape[0])
    #print (r_imputed_training_data[1:5,8], r_imputed_testing_data[1:5,8])
    print ('='*20)

    # ## use median imputer
    m_imputer = Imputer(axis=0, missing_values='NaN', strategy='median', verbose=0)  # median deletes first column
    m_imputed_training_data = m_imputer.fit_transform(np.copy(cleaned_training_data))
    m_imputed_testing_data = m_imputer.transform(np.copy(cleaned_testing_data))
    pickle.dump(m_imputer, open(params['m_imputer_out'] + '.pkl', 'wb'))
    np.save(params['assemble_output_filename'] + '_mImp', m_imputed_training_data)
    np.save(params['assemble_output_filename'] + '_mImp_names', cleaned_training_names)
    np.save(params['assemble_eval_output_filename'] + '_mImp', m_imputed_testing_data)
    np.save(params['assemble_eval_output_filename'] + '_mImp_names', cleaned_testing_names)

    print ('Train mImp: ', m_imputed_training_data.shape, cleaned_training_names.shape, m_imputed_training_data.shape[1] == cleaned_training_names.shape[0])
    print ('Eval mImp: ', m_imputed_testing_data.shape, cleaned_testing_names.shape, m_imputed_testing_data.shape[1] == cleaned_testing_names.shape[0])
    #print (m_imputed_training_data[1:5,8], m_imputed_testing_data[1:5,8])
    print ('+'*20)

    return


def determine_feature_matrix_and_target_matrix_rows(params):
    """
    This generates a file that stores the followg details:
    feature_matrix_name, target_id, target_name, fold_type [all, 0, 1, 2, 3, 4], [row indices for desired samples]
    The row indices are based on case order rows and item_present-labeling

    It also generates a file that stores the followg details:
    target_matrix_name, target_id, fold_type [all, 0, 1, 2, 3, 4], [row indices for desired samples]
    The row indices are based on target order rows and the rows used in the above file

    It also generates a file that stores the following details:
    feature_matrix_name, target_id, fold_type [all, 0, 1, 2, 3, 4], [case ids of desired samples]
    The row indices are based on the two files above
    """
    def load_target_present_rows(item_present_file, target_name):
        """
        This function determines which cases (samples) a target was present for.
        """
        target_present_rows = []
        with open(item_present_file, 'r') as f:
            target_name_file_column_idx = False
            first_line = True
            for full_line in f:
                line = full_line.rstrip()
                if not line:  # insure line is not empty
                    break
                split_line = full_line.rstrip().split(',')
                if first_line:  # first line
                    first_line = False
                    target_name_file_column_idx = split_line.index(target_name)
                elif int(split_line[target_name_file_column_idx]):  # check if column is '1' for current row
                    target_present_rows.append(split_line[1])  # add case_id
        return target_present_rows

    case_order_rows = load_list(params['case_order_rows_file'])  # the case order in the feature matrix
    target_case_rows = load_list(params['target_case_rows_file'])  # the case order in the target matrix
    target_feature_columns = load_list(params['target_feature_columns_file'])  # the target names for each column in the target matrix

    feature_samples_outfile = open(params['feature_samples_outfile'], 'w')
    target_samples_outfile = open(params['target_samples_outfile'], 'w')
    feat_and_targ_samples_outfile = open(params['feat_targ_samples_outfile'], 'w')
    feature_samples_outfile.write('#feature_matrix_name, target_id, target_name, fold_type, row_indices\n')
    target_samples_outfile.write('#target_matrix_name, target_id, target_name, fold_type, row_indices\n')
    feat_and_targ_samples_outfile.write('#target_matrix_name, target_id, target_name, fold_type, case_ids\n')
    
    for t_idx, target_name in enumerate(target_feature_columns):
        target_present_rows = load_target_present_rows(parmas['item_present_file'], target_name)
        samples_full_fold_type = []
        target_full_fold_type = []
        feat_targ_full_fold_type = []
        sam_folds = [[] for x in range(5)]  # five folded
        tar_folds = [[] for x in range(5)]  # five folded
        feat_folds = [[] for x in range(5)]  # five filded
        count = 0
        for idx, case_id in enumerate(case_order_rows):
            if case_id in target_present_rows:
                samples_full_fold_type.append(str(idx))
                target_full_fold_type.append(str(target_case_rows.index(case_id)))
                feat_targ_full_fold_type.append(case_id)
                sam_folds[count%5].append(str(idx))
                tar_folds[count%5].append(str(target_case_rows.index(case_id)))
                feat_folds[count%5].append(case_id)
                count += 1

        # ## print the full fold type
        feature_samples_outfile.write(params['feature_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+'full'+'\t'+'\t'.join(samples_full_fold_type)+'\n')
        target_samples_outfile.write(params['target_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+'full'+'\t'+'\t'.join(target_full_fold_type)+'\n')
        feat_and_targ_samples_outfile.write(params['target_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+'full'+'\t'+'\t'.join(feat_targ_full_fold_type)+'\n')

        # ## print for the five folds
        for f_idx in range(5):
            feature_samples_outfile.write(params['feature_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+str(f_idx)+'\t'+'\t'.join(sam_folds[f_idx])+'\n')
            target_samples_outfile.write(params['target_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+str(f_idx)+'\t'+'\t'.join(tar_folds[f_idx])+'\n')
            feat_and_targ_samples_outfile.write(params['target_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+str(f_idx)+'\t'+'\t'.join(feat_folds[f_idx])+'\n')

    feature_samples_outfile.close()
    target_samples_outfile.close()
    feat_and_targ_samples_outfile.close()

    return


def populate_imputation_params(base_dir):
    """
    Populates parameter dictionary.

    # some params may not be used 
    """
    params = {}
    params['assemble_output_filename'] = base_dir + 'feature_matrix_storage_labeling_cases/full_labeling'
    params['assemble_eval_output_filename'] = base_dir + 'feature_matrix_storage_evaluation_cases/full_evaluation'
    params['keep_columns_out'] = base_dir + 'imputer_storage/keep_col_imputer-full_labeling.pkl'
    params['r_imputer_out'] = base_dir + 'imputer_storage/r_imputer-full_labeling'
    params['m_imputer_out'] = base_dir + 'imputer_storage/m_imputer-full_labeling'
    params['unit_testing'] = False
    return params


def populate_sample_rows_params(base_dir):
    """
    Populates parameter dictionary.

    # some params may not be used 
    """
    params = {}
    case_dir = base_dir + 'complete_feature_files_labeling_cases/'
    out_dir = base_dir + 'feature_matrix_storage_labeling_cases/'
    params['case_order_rows_file'] = case_dir + 'case_order_rows.txt'
    params['target_case_rows_file'] = case_dir + 'target_case_rows.txt'
    params['target_feature_columns_file'] = case_dir + 'target_feature_columns.txt'
    params['feature_samples_outfile'] = out_dir + 'feature_samples_out.txt'
    params['target_samples_outfile'] = out_dir + 'target_samples_out.txt'
    params['feat_targ_samples_outfile'] = out_dir + 'feat_targ_samples_out.txt'
    params['item_present_file'] = case_dir + 'items_present-labeling.txt'
    params['feature_matrix_name'] = out_dir + 'full_labeling_'
    params['target_matrix_name'] = case_dir + 'target_full_matrix'
    return params


if __name__=="__main__":

    # clean columns and impute data
    params = populate_imputation_params('/my_base_dir/') 
    imputation(params)                            

    # ## select target columns and sample rows
    params = populate_sample_rows_params('/my_base_dir/')
    determine_feature_matrix_and_target_matrix_rows(params)
