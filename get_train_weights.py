# function for getting the training weights - use get_weights(patients_to_take_all_train)
# use check_weights to check two important properties hold
# considers both the class weights (weights smaller classes higher) and patient weights (weights all patients equally within classes, so multiple samples from the same patient get weighted lower)
# the same functions are in determining_CV_split_and_weights.ipynb
import numpy as np
import sklearn
import pandas as pd

def get_patient_weights(patients_to_take_all_train, c):
    # can use the same sklearn function, and work out within each class:
    patient_ids_in_class = patients_to_take_all_train[patients_to_take_all_train['pooled_labels'] == c]['patient_id']
    patient_ids_in_class = patient_ids_in_class.astype('str').str.replace('^X', '1').astype('int') # removing 'X's, prefixing a 1 to make unique from the other ints and turning to int

    weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(patient_ids_in_class), y=patient_ids_in_class)
    weights_df = pd.DataFrame(weights, np.unique(patient_ids_in_class))
    weights_df['patient_id'] = weights_df.index

    # check weights of patients - sum all of a patient's samples
    print('should be 1 number in list: ', weights_df.loc[patient_ids_in_class].groupby('patient_id').sum()[0].unique()) # should be just 1 number

    return(weights_df.loc[patient_ids_in_class])


def add_combined_weights(patients_to_take_all_train, c, class_weights):
    to_add = get_patient_weights(patients_to_take_all_train, c)
    to_add.columns = [c, 'patient_id']
    to_add
    to_add['patient_id'] = to_add.index
    to_add['pooled_labels'] = c
    
    to_add = to_add.drop_duplicates() # remove now unneeded duplicates 
    
    # to match to_add:
    patients_to_take_all_train['patient_id'] = patients_to_take_all_train['patient_id'].astype('str').str.replace('^X', '1').astype('int')
    
    # putting the weights for each sample in the dataframe:
    res = pd.merge(patients_to_take_all_train, to_add, how = 'left', on = ['pooled_labels', 'patient_id'])
    
    # multiplying patient weights by class weights (nans should stay unaffected)
    res[c] = res[c]*class_weights[c]
    
    return(res)
    


def get_weights(patients_to_take_all_train):
    y_train = np.array(patients_to_take_all_train['pooled_labels'])
    
    # weights to deal with imbalanced classes:
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    # weights for patients:
    # add weights for all classes separately
    res = pd.concat([add_combined_weights(patients_to_take_all_train, 0, class_weights), 
               add_combined_weights(patients_to_take_all_train, 1, class_weights), 
               add_combined_weights(patients_to_take_all_train, 2, class_weights),
               add_combined_weights(patients_to_take_all_train, 3, class_weights), 
               add_combined_weights(patients_to_take_all_train, 4, class_weights), 
               ], axis = 1)

    # checking duplicate column names are actually the same, so we can remove them
    a = np.array(res['pooled_labels'].transpose())
    print(np.all(a == a[0,:], axis = 1))
    a = np.array(res['patient_id'].transpose())
    print(np.all(a == a[0,:], axis = 1))

    res = res.loc[:,~res.columns.duplicated()]

    # combining the 5 columns for each class into 1 weight column. The nanmean function removes nans and keeps the floats
    c_1 = np.nanmean([res[0], res[1]], axis = 0)
    c_2 = np.nanmean([c_1, res[2]], axis = 0)
    c_3 = np.nanmean([c_2, res[3]], axis = 0)
    c_4 = np.nanmean([c_3, res[4]], axis = 0)

    res['combined_weight'] = c_4

    # check right order
    print(np.all(np.array(res['pooled_labels']) == np.array(patients_to_take_all_train['pooled_labels']))) # should be True
    print(np.all(np.array(res['patient_id']) == np.array(patients_to_take_all_train['patient_id']))) # should be True
    
    return(res[['pooled_labels', 'patient_id', 'combined_weight']])


# checking weights - 
# want patients to be total same weight in each class
# want each class to contribute same amount, so total class weights should be equal
def check_weights(train_weights): # uses output from get_weights()
    
    # for each class, what is the total weight (summed across all their samples) for each patient?
    print('should be 1 number per class:')
    print([train_weights[train_weights['pooled_labels'] == i].groupby('patient_id').sum()['combined_weight'].unique() for i in range(5)])
    # yes these are all equal for each class (aside for some rounding issues)


    # what is the total weight of each class?
    print('should be the same number for each class:')
    print(train_weights[['pooled_labels', 'combined_weight']].groupby('pooled_labels').sum()) # good, it is equal

