from sklearn import metrics

# given a confusion matrix, plots it so it looks nice
def plot_confusion_matrix(conf_mat, model_type, save_path = None, font_size = 36, cm_labels = []):
    import seaborn as sns
    
    sns.set(font_scale=1.75)
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.rcParams.update({'font.size': font_size})
        
    plot = sns.heatmap(conf_mat, annot=True, fmt='d', square = True, cbar = False, cmap='Blues', linewidths=2, linecolor="black", annot_kws = {"size": 20}, xticklabels = cm_labels, yticklabels = cm_labels)
    plot.set(xlabel='Predicted label', ylabel='True label')
    
    fig = plot.get_figure()
    if save_path == None:
        fig.savefig('figs/confusion_matrix_'+model_type+'.svg', bbox_inches='tight')
    else:
        fig.savefig(save_path,  bbox_inches='tight')
    



def print_evaluation(fitted, m_values_test, diagnoses_test, model_type, predictions=None, prob_predictions=[], no_roc = False, cm_labels = [], save_folder = ''):
    
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 26})

    if fitted != None: # if a fitted, find predictions (else use predictions)
        predictions = fitted.predict(m_values_test)
    
    accuracy = metrics.accuracy_score(diagnoses_test, predictions)
    
    conf_mat = metrics.confusion_matrix(diagnoses_test, predictions)
    
    precision = metrics.precision_score(diagnoses_test, predictions, average = None)
    recall = metrics.recall_score(diagnoses_test, predictions, average = None)
    f1 = metrics.f1_score(diagnoses_test, predictions, average = None)
    mcc = metrics.matthews_corrcoef(diagnoses_test, predictions) # matthews correlation coefficient is supposed to be better at dealing with class imbalance than f1
    # NOTE: with multiple classes, mcc does not have a minimum value of -1 (it will be between -1 and 0). The max value is always 1
    roc_auc = None
    
    #  for roc_auc we need a one hot format: (only works if we have fitted or prob_predictions)
    try:
        if fitted != None and no_roc == False:
            import pandas as pd
            diagnoses_one_hot = pd.get_dummies(pd.Series(diagnoses_test))
            predictions_one_hot = fitted.predict_proba(m_values_test)
            roc_auc = metrics.roc_auc_score(diagnoses_one_hot, predictions_one_hot, average = None)
    except:
        print("Couldn't calculate roc_auc")
        
    try:
        if prob_predictions != [] and no_roc == False:
            import pandas as pd
            diagnoses_one_hot = pd.get_dummies(pd.Series(diagnoses_test))        
            roc_auc = metrics.roc_auc_score(diagnoses_one_hot, prob_predictions, average = None)
    except:
        print("Couldn't calculate roc_auc")

    print("Acc, conf mat:")
    print(accuracy)
    print(conf_mat)
    print("precision, recall, f1 for each class:")
    print(precision, recall, f1)
    print("matthews correlation coeficient")
    print(mcc)
    if fitted != None or prob_predictions != []:
        print("roc_auc for each class: ", roc_auc)
        
    plot_confusion_matrix(conf_mat, model_type, save_path = save_folder + 'confusion_matrix_'+model_type+'.svg', cm_labels = cm_labels)
    
    import pandas as pd
    model_metrics = pd.DataFrame(data = {'name': ["Accuracy", "Precision", "Recall", "f1", "mcc", "roc_auc"], 'values': [accuracy, precision, recall, f1, mcc, roc_auc]})
    
    model_metrics.to_csv(save_folder + "metrics_" + model_type + ".csv", index = False)
    
    return predictions






from sklearn.metrics import roc_curve, precision_recall_curve, auc

# can plot ROC curves (curve_type='roc') or precision recall curves (curve_type='precision_recall')
# plot_curve('roc', diagnoses, probs, len(labels), labels, classification_type, save_folder = 'figs_xgboost/')
def plot_curve(curve_type, diagnoses, probs, num_classes, classes, colours, classification_type, save_folder = ''):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 30})
    plt.clf()
    
    if curve_type == 'roc':
        stats_per_class = [roc_curve(diagnoses, probs[:,class_num], pos_label = class_num) for class_num in range(num_classes)]
        xs = [stats_per_class[i][0] for i in range(num_classes)]
        ys = [stats_per_class[i][1] for i in range(num_classes)]
        
        # now calc AUC scores
        import pandas as pd
        diagnoses_one_hot = pd.get_dummies(pd.Series(diagnoses))
        
        roc_auc = metrics.roc_auc_score(diagnoses_one_hot, probs, average = None)
        
    elif curve_type == 'precision_recall':
        stats_per_class = [precision_recall_curve(diagnoses, probs[:, class_num], pos_label = class_num) for class_num in range(num_classes)]
        xs = [stats_per_class[i][1] for i in range(num_classes)] # recall is x
        ys = [stats_per_class[i][0] for i in range(num_classes)] # precision is y
        
        pr_aucs = [auc(xs[i], ys[i]) for i in range(len(xs))]
    else:
        print("Curve type ", curve_type, " not known.")
        return 1
 
    # using seaborn
    import seaborn as sns
    
    # create dataframe that includes tpr and fpr info for all classes (with each class having its own column):
    import pandas as pd
    data = pd.DataFrame(ys[0], xs[0], columns=[0])
    for i in range(1, num_classes):
        data = data.append(pd.DataFrame(ys[i], xs[i], columns=[i]))
    data.columns = classes
    
    # plotting
    
    lw = 2
    
    sns.set(font_scale=1.5, style="ticks")
    
    palette = {k:v for k,v in zip(classes, colours)}
    
    if curve_type == 'roc':
        plot = sns.lineplot(data = data, linewidth=lw, legend="full", estimator=None, dashes=False, palette=palette, ci = None) 
    elif curve_type == 'precision_recall':
        plot = sns.lineplot(data = data, linewidth=lw, legend="full", dashes=False, palette=palette, ci = None) 
       
    plt.ylim(0, None) # make sure y axis starts at 0
    
    import numpy as np
    if curve_type == 'roc':
        labels = [classes[i] + ' (AUC = ' + str(np.round(roc_auc[i], 3)) + ')' for i in range(len(classes))]
        plt.legend(title='Class', bbox_to_anchor=(0.98, 0.05), loc='lower right', borderaxespad=0., labels=labels)
    elif curve_type == 'precision_recall':
        labels = [classes[i] + ' (AUC = ' + str(np.round(pr_aucs[i], 3)) + ')' for i in range(len(classes))]
        plt.legend(title='Class', bbox_to_anchor=(1.1, 0.05), loc='lower left', borderaxespad=0., labels=labels)
    

    if curve_type == 'roc':
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
    elif curve_type == 'precision_recall':
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    # save
    fig = plot.get_figure()
    fig.savefig(save_folder + curve_type+'_curve_'+classification_type+'.svg', bbox_inches = 'tight')

    if curve_type == 'roc':
        return roc_auc
    elif curve_type == 'precision_recall':
        return pr_aucs