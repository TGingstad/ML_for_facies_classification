import pandas as pd
import matplotlib.pyplot as plt
import plotly as px
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../ML_for_facies_classification/functions')

from segy_file_read_plot import segy_file
from plot_2D_sections import plot_2D_section, difference_map
from data_processing import replace_nonexisting_data_with_NaN, take_data_that_exist, df_wells_from_section, standartization
from facies_features_concat import concat_features_Any
from machine_learning import predict_2d_Any
from machine_learning import accuracy_score_cv, confusion_matrix_prediction, feature_importance_plot


class ML_classification:

    def __init__(self, features=[], well_line = '', facies = '', resolution = ''):
        self.features = features
        self.well_line = well_line
        self.Facies = facies
        self.resolution = resolution
        self.number_features = len(features)

    def read_segy(self, image=True):
        
        self.features_variables=[]
        self.features_variables_extent=[]
        self.features_variables_well=[]
        self.features_variables_extent_well=[]

        if self.Facies == 'simple':
            color = 'facies_simple'
        else:
            color = 'facies'

        # Load Facies 2D line
        feature , extent_feature = segy_file(
            file_name=f'../ML_for_facies_classification/'
            f'segy_files/standard/Facies_type/{self.Facies}/Facies', color_map=color, image=image)
        self.features_variables.append(feature)
        self.features_variables_extent.append(extent_feature)

        # Load Facies wells
        feature , extent_feature = segy_file(
            file_name=f'../ML_for_facies_classification/'
            f'segy_files/standard/Facies_type/{self.Facies}/{self.well_line}/W_Facies', color_map=color, image=image)
        self.features_variables_well.append(feature)
        self.features_variables_extent_well.append(extent_feature)

        # Load seismic and attributes
        for i in range(self.number_features):

            if self.features[i] == 'Seismic_p' or 'Seismic_s':
                color = 'seismic'
            else:
                color = 'any'

            # Load seismic 2D line
            feature , extent_feature = segy_file(file_name=f'../ML_for_facies_classification/'
                                                 f'segy_files/standard/{self.resolution}/{self.features[i]}', color_map=color, image=image)
            self.features_variables.append(feature)
            self.features_variables_extent.append(extent_feature)

            # Load seismic wells
            feature , extent_feature = segy_file(file_name=f'../ML_for_facies_classification/'
                                                 f'segy_files/standard/{self.resolution}/{self.well_line}/W_{self.features[i]}', color_map=color, image=image)
            self.features_variables_well.append(feature)
            self.features_variables_extent_well.append(extent_feature)

    
    def segy_to_df_and_modify(self):
        # Extract all data from SEGy to DataFrame
        self.df_2d_line = []
        self.df_2d_well_line = []
        for i in range(self.number_features + 1):
            # Convert to dataframe
            self.df_2d_line.append(pd.DataFrame(self.features_variables[i].T))
            self.df_2d_well_line.append(pd.DataFrame(self.features_variables_well[i].T))

            # modify/process dataframe
            self.df_2d_line[i] = self.df_2d_line[i].iloc[0:701]
            self.df_2d_well_line[i] = self.df_2d_well_line[i].iloc[0:701]

            # Remove data outside reservoir
            self.df_2d_line[i] = replace_nonexisting_data_with_NaN(self.df_2d_line[i], self.df_2d_line[i].iloc[0,0])
            self.df_2d_well_line[i] = replace_nonexisting_data_with_NaN(self.df_2d_well_line[i], self.df_2d_well_line[i].iloc[0,0])


    def check_2d_sections(self):

        if self.Facies == 'simple':
            color = 'facies_simple'
        else:
            color = 'facies'

        # Plot 2d facies
        plot_2D_section(data_file=self.df_2d_line[0].T, extent_plot=self.features_variables_extent[0], 
                        color_map=color, number_of_facies=5, list_of_wells=None)
        plot_2D_section(data_file=self.df_2d_well_line[0].T, extent_plot=self.features_variables_extent_well[0], 
                        color_map=color, number_of_facies=5, list_of_wells=None)

        # Plot seismic and attributes
        for i in range(self.number_features):
            if self.features[i] == 'Seismic_p' or 'Seismic_s':
                color = 'seismic'
                plot_2D_section(data_file=self.df_2d_line[i+1].T, extent_plot=self.features_variables_extent[i+1], 
                                color_map=color, number_of_facies=5, list_of_wells=None)
                plot_2D_section(data_file=self.df_2d_well_line[i+1].T, extent_plot=self.features_variables_extent_well[i+1],
                                color_map=color, number_of_facies=5, list_of_wells=None)

            else:
                color = 'any'
                plot_2D_section(data_file=self.df_2d_line[i+1], extent_plot=self.features_variables_extent[i+1], 
                                color_map=color, number_of_facies=5, list_of_wells=None)
                plot_2D_section(data_file=self.df_2d_well_line[i+1], extent_plot=self.features_variables_extent_well[i+1], 
                                color_map=color, number_of_facies=5, list_of_wells=None)


    def statistics(self):
        # Seismic line
        print('seismic line:')
        fig = px.histogram(self.df_2d_line[0].stack(), width=600, height=400, histnorm='percent', text_auto='.1f')
        fig.update_layout(yaxis_title="Facies' percentage, %")
        fig.update_layout(xaxis_title="Facies code")
        fig.update_layout(yaxis_range=[0,100])
        fig.update_layout(bargap=0.2)
        fig.show()

        wellss = self.df_2d_well_line[0]
        # Wells
        print('well line:')
        fig = px.histogram(wellss[self.col].stack(), width=600, height=400, histnorm='percent', text_auto='.1f')
        fig.update_layout(yaxis_title="Facies' percentage, %")
        fig.update_layout(xaxis_title="Facies code")
        fig.update_layout(yaxis_range=[0,100])
        fig.update_layout(bargap=0.2)
        fig.show()

    def standartization(self):
        self.df_2d_line_std = []
        self.df_2d_well_line_std = []

        for i in range(self.number_features):
            self.df_2d_line_std.append(standartization(self.df_2d_line[i+1], value_to_drop=False))
            self.df_2d_well_line_std.append(standartization(self.df_2d_well_line[i+1], value_to_drop=False))
    
    def create_wells(self):
        # Create a list of wells
        if self.well_line == 'W0':
            self.col = [1, 192, 451]
        
        elif self.well_line == 'W1':
            self.col = [1, 225, 319]
        
        elif self.well_line == 'W2':
            self.col = [1, 94, 300]

        elif self.well_line == 'W3':
            self.col = [1, 49, 192]
        
        print(self.df_2d_well_line[0][self.col].apply(pd.Series.value_counts)) 

        self.df_wells = []
        self.df_wells.append(df_wells_from_section(self.df_2d_well_line[0], self.col))

        for i in range(self.number_features):
            self.df_wells.append(df_wells_from_section(self.df_2d_well_line_std[i], self.col))
        
        
    def concatenate_facies_features(self):

        self.facies_features = concat_features_Any(self.df_wells[0], self.df_wells[1:], self.features) 
        print(self.facies_features['facies'].value_counts()) 


    def train_RF_model(self):
        # Divide on train and validation set

        feature_list = []
        for i in range(len(self.features)):
            feature_list.append(f'{self.features[i]}')
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.facies_features[feature_list], 
                                                            self.facies_features[['facies']], 
                                                            train_size=0.8,
                                                            random_state=123)
        

    def RF_classifier(self):
        self.RF_Classifier_model = RandomForestClassifier(class_weight='balanced').fit(self.x_train, self.y_train)
        facies_predict_RF = self.RF_Classifier_model.predict(self.x_test)

        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
        print(confusion_matrix(self.y_test, facies_predict_RF))
        print(accuracy_score(self.y_test, facies_predict_RF))
        print(classification_report(self.y_test, facies_predict_RF))

        accuracy_score_RF_7_train = accuracy_score_cv(estimator=self.RF_Classifier_model, X=self.x_train, y=self.y_train, cv=10)
        accuracy_score_RF_7_train

        accuracy_score_RF_7_test = accuracy_score_cv(estimator=self.RF_Classifier_model, X=self.x_test, y=self.y_test, cv=10)
        accuracy_score_RF_7_test

        
        self.facies_pred, self.df_facies_comparison = predict_2d_Any(df_facies=self.df_2d_line[0], 
                                                                  feature_list = self.df_2d_line_std,                                                                        
                                                                  model=self.RF_Classifier_model)
       
        
    def results(self):

        if self.Facies == 'simple':
            color = 'facies_simple'
        else:
            color = 'facies'

        plot_2D_section(data_file=self.facies_pred.T, extent_plot=self.features_variables_extent[0], 
                        color_map=color, number_of_facies=5, list_of_wells=None)
        
        facies_class = ['Sand', 'Fine Sand', 'Coarse Sand', 'Shale', 'Carbonate']

        if self.Facies == 'simple':
            facies_class = ['Sand', 'Shale', 'Carbonate']

        self.report_print_RF_7, self.f1_score_per_class_RF_7, self.count_facies, self.accuracy_test_RF_7  = confusion_matrix_prediction(
                                                                                                        self.df_facies_comparison, 
                                                                                                        self.facies_pred, 
                                                                                                        col_number=self.col, 
                                                                                                        facies_class=facies_class)
        
        difference_map(df_facies_comparison=self.df_facies_comparison, facies_predicted=self.facies_pred, 
                       extent=self.features_variables_extent[0], list_of_wells=None)


    def result_plots(self):
        # Plot number of facies VS F1_score
        facies_class = ['Sand', 'Fine Sand','Coarse Sand', 'Shale', 'Carbonate']

        if self.Facies == 'simple':
            facies_class = ['Sand', 'Shale', 'Carbonate']

        fig, ax = plt.subplots()
        ax.scatter(self.count_facies, self.f1_score_per_class_RF_7)

        for i, txt in enumerate(facies_class):
            ax.annotate(txt, (self.count_facies[i], self.f1_score_per_class_RF_7[i]))
    
        plt.ylabel('F1-score')
        plt.xlabel('Number of Fcaies')
        plt.show()

        from machine_learning import feature_importance_plot
        feature_importance_plot(self.RF_Classifier_model, self.x_train, self.y_train, random_state=50)

    
    def run_results(self):
        self.read_segy(image=False)
        self.segy_to_df_and_modify()
        #self.check_2d_sections()
        self.standartization()
        self.create_wells()
        #self.statistics()
        self.concatenate_facies_features()
        self.train_RF_model()
        self.RF_classifier()
        self.results()
        self.result_plots()