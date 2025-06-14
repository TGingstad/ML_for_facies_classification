from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def model_LR(facies, features):
    """_summary_

    Args:
        facies (DataFrame): DataFrame with facies. Defaults to facies_and_features[['facies']].
        features (DataFrame): DataFrame with features. Defaults to facies_and_features[['relai', 'seis']].

    Returns:
        x_train, x_test, y_train, y_test (DataFrame): data used for training and validation
    """
    
    # Train-validation split
    x_train, x_test, y_train, y_test = train_test_split(features, facies, train_size=0.8, random_state=123)
    # return  x_train, x_test, y_train, y_test

    model_LR = LogisticRegression()
    model_LR.fit(x_train, y_train)  
    test_predict = model_LR.predict(x_test)
    
    return model_LR

#######################################################################

def predict_2d_Any(df_facies, feature_list, model):
    """This function uses ML model and attributes to predict 2D facies cross-section
    It takes the shape of the ground-truth facies and predicts facies using 
    specified attributes (relai, rms, D2, etc...)

    Args:
        df_facies (DataFrame): facies cross-section
        feature_list (list): list of 2D cross-section DataFrames of input attributes 
        model: ML model

    Returns:
        map_facies_1 (array) : the predicted facies
        df_f_comparison (array): the filtered actual facies for comparison
    """

    # create empty Numpy array with the same shape as facies
    r, c = np.shape(df_facies)
    map_facies = np.zeros((r, c))
    map_facies[:] = np.NaN
    map_facies_1 = map_facies.copy()

    # create array for facies filter
    df_f_comparison = map_facies.copy()

    # for every trace (column)
    for every_col in range(len(df_facies.columns)):
        feature_columns = []
        for i in range(len(feature_list)):

            feature_columns.append(feature_list[i].iloc[:, every_col])

        # concat every column
        new_df = pd.concat(feature_columns, axis=1)

        # remove NaN values from trace
        new_df_1 = new_df.dropna(axis=0)

        # remove NaN values from Facies trace
        non_empty_facies = df_facies.iloc[:, every_col].dropna(axis=0)

        # predict
        map_facies[new_df_1.index, every_col] = model.predict(new_df_1)

        # Filter Facies to compare
        map_facies_1[non_empty_facies.index, every_col] = map_facies[non_empty_facies.index, every_col]
        df_f_comparison[new_df_1.index, every_col] = df_facies.iloc[new_df_1.index, every_col]

    return map_facies_1, df_f_comparison
    

###########################################################################################################
    
def accuracy_score_cv(estimator, X, y, cv=10):
    
    from sklearn.model_selection import cross_val_score
    #Applying 10-fold cross validation
    accuracy_score_cv = cross_val_score(estimator=estimator, X=X, y=y, cv=cv)
    print("accuracy: ", np.mean(accuracy_score_cv))
    
    return np.mean(accuracy_score_cv)


# Feature importance
def feature_importance_plot(model, x_train, y_train, random_state):

    from sklearn.inspection import permutation_importance
    
    res = permutation_importance(model, x_train, y_train, scoring='accuracy', random_state=random_state)
    importance = res.importances_mean
    importance
    importance_res = pd.Series(importance * 100, index=x_train.columns).sort_values(ascending=True)
    importance_res
    # Plot the results
    fig, ax = plt.subplots(figsize=(8,4))
    ax = importance_res.plot.barh()
    ax.set_title('Permutation importance', fontsize=14)
    ax.set_ylabel('Importance score', fontsize=14)
    ax.set_xlabel('Percentage, %', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()
    
    
# Remove data that were used for training
def confusion_matrix_prediction(df_facies_comparison, facies_pred, col_number, facies_class): 
       
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
        
    df_f_comparison_pd = pd.DataFrame(df_facies_comparison)

    map_facies_pd = pd.DataFrame(facies_pred)

    actual_f = df_f_comparison_pd.values[~(np.isnan(df_f_comparison_pd))]
    predicted_f = map_facies_pd.values[~(np.isnan(map_facies_pd))]

    conf_matrix = metrics.confusion_matrix(actual_f, predicted_f)
    conf_matrix
    print(pd.crosstab(actual_f, predicted_f))
    report_print = print(classification_report(actual_f, predicted_f))
    
    f1_score_per_class = f1_score(actual_f, predicted_f, average=None)
    accuracy_estimation = accuracy_score(actual_f, predicted_f)
    
    # Extract number of values of each class
    count_facies = np.unique(actual_f, return_counts=True)[1]
    
    display_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.array(facies_class))
    display_conf_matrix.plot()
    plt.show()
    
    return report_print, f1_score_per_class, count_facies, accuracy_estimation