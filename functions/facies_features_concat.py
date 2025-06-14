import pandas as pd

def concat_features_Any(df_facies_wells, feature_list, features_names):
    """The function concatenate any number of features (Seismic, RelAcImp, Rms, etc..) with Facies

    Args:
        df_facies_wells (DataFrame): Facies
        feature_list (list): list of features
        feature_names (list): name of features 
        
    Returns:
        facies_and_features (DataFrame): concatenated dataframe with facies and features 
    """

    # Create emty list of lists
    list_features = []
    for i in range(len(feature_list)):
        list_features.append([])

    list_facies = []
    concated_features = []

    for every_well in range(len(df_facies_wells.columns)):
        for i in range(len(feature_list)):

            list_features[i].append(feature_list[i].iloc[:, every_well])

        list_facies.append(df_facies_wells.iloc[:, every_well])
    
    for i in range(len(feature_list)):
        concated_features.append(pd.concat(list_features[i], ignore_index=False))

    facies = pd.concat(list_facies, ignore_index=False)

    # Transform to dataframe
    features_df = pd.concat(concated_features, axis=1)
    facies_df = pd.DataFrame(facies).rename(columns={0:'facies'})

    # Rename features
    for i in range(len(feature_list)):

        features_df = features_df.rename(columns = {i: f'{features_names[i]}'})

    # Concate facies and features
    facies_and_features = pd.concat([facies_df, features_df], axis=1)
    facies_and_features = facies_and_features.dropna(axis=0)

    return facies_and_features