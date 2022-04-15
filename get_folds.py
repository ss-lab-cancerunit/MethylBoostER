# functions to get stratified folds that do not have any patient overlap
# also checks these folds and plots the class distributions

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd

# returns stratified folds in the form of: [(train_indices, test_indices), ...]
# ensures that no patients overlap between folds
def get_folds(diagnoses, num_folds):
    # get unique patients
    unique_patients = diagnoses[['pooled_labels', 'patient_id']].drop_duplicates()
    # remove normal EPIC samples: (as we want them to be in the same split as the tumour samples)
    unique_patients = unique_patients[~((unique_patients['patient_id'].str.contains('X', na = False)) & (unique_patients['pooled_labels'] == 0))]

    # make folds for the patients
    skf = StratifiedKFold(n_splits=num_folds, shuffle = True)
    patient_folds = list(skf.split(np.zeros(len(unique_patients['pooled_labels'])), unique_patients['pooled_labels'])) # we don't need to use data, the docs say we can replace it with np.zeros()

    result_folds = []
    # now get sample indices for each patient
    for train_patients, test_patients in patient_folds:
        train_patient_ids = unique_patients.iloc[train_patients]['patient_id']
        train_diagnoses_indices = np.where(diagnoses['patient_id'].isin(train_patient_ids))[0]
        # test indices are the complement of the train indices:
        test_diagnoses_indices = np.asarray(list(set(range(len(diagnoses['patient_id']))) - set(train_diagnoses_indices)))

        result_folds.append((train_diagnoses_indices, test_diagnoses_indices))

    return(result_folds)


# checks there is no patient overlap in folds
def check_folds(diagnoses, folds):
    print('Should be empty sets if no patient overlaps:')
    for train_indices, test_indices in folds:
        train_patients = set(diagnoses.iloc[train_indices]['patient_id'])
        test_patients = set(diagnoses.iloc[test_indices]['patient_id'])
        print(train_patients.intersection(test_patients))



# plot the number of samples in each class for the given split (both train and test plots). Also plot the total number in train and test
def plot_split_barplots(labels, folds):  

    train_labels = [labels[train_indices] for train_indices, _ in folds]
    test_labels = [labels[test_indices] for _, test_indices in folds]

    plt.figure(figsize = (5, 4))
    
    counts = [np.unique(train_labels[i], return_counts=True)[1] for i in range(4)] # for each fold, how many samples in each class?
    to_plot = pd.DataFrame(counts).melt() # melt puts it in long form
    to_plot['fold'] = np.tile(list(range(4)), 5)
    sb.barplot(data = to_plot, x = 'variable', y = 'value', hue = 'fold')
    plt.title('Train counts per fold')
    plt.xlabel('Class')
    plt.legend('')
    plt.show()

    plt.figure(figsize = (5, 4))
    counts = [np.unique(test_labels[i], return_counts=True)[1] for i in range(4)] # for each fold, how many samples in each class?
    to_plot = pd.DataFrame(counts).melt() # melt puts it in long form
    to_plot['fold'] = np.tile(list(range(4)), 5)
    sb.barplot(data = to_plot, x = 'variable', y = 'value', hue = 'fold')
    plt.title('Test counts per fold')
    plt.xlabel('Class')
    plt.legend('')
    plt.show()
    # both plots should have the same height for each class (variable)


    # now plotting the group sizes:
    to_plot = pd.DataFrame({'sizes': [len(train_labels[i]) for i in range(4)] + [len(test_labels[i]) for i in range(4)], 
                            'type': list(np.repeat('train', 4)) + list(np.repeat('test', 4)),
                            'fold': np.tile(range(4), 2)})

    plt.figure(figsize = (5, 4))
    
    sb.barplot(data = to_plot, x = 'type', y= 'sizes', hue = 'fold')
    plt.legend('')
    plt.show()


