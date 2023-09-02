'''
This script aims to provide functions that will turn the exploratory data analysis (EDA) process easier. 
'''


'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')


def sns_plots(data, features, histplot=True, countplot=False,     
              barplot=False, barplot_y=None, boxplot=False, 
              boxplot_x=None, outliers=False, kde=False, 
              hue=None):
    """
    Generate a grid of Seaborn plots for visualizing multiple features of a dataset.

    Args:
        data (DataFrame): The dataset to visualize.
        features (list): List of feature names to visualize.
        histplot (bool, optional): If True, generate histogram plots. Default is True.
        countplot (bool, optional): If True, generate count plots. Default is False.
        barplot (bool, optional): If True, generate bar plots. Default is False.
        barplot_y (str, optional): The y feature for bar plots. Required if barplot is True.
        boxplot (bool, optional): If True, generate box plots. Default is False.
        boxplot_x (str, optional): The x feature for box plots. Required if boxplot is True.
        outliers (bool, optional): If True, show outliers in box plots. Default is False.
        kde (bool, optional): If True, show kernel density estimate in histogram plots. Default is False.
        hue (str, optional): Feature to group plots by color (hue). Default is None.

    Returns:
        None
    """
    
    try:
        num_features = len(features)
        num_rows = num_features // 3 + (num_features % 3 > 0)  

        fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5*num_rows))  

        for i, feature in enumerate(features):
            row = i // 3  
            col = i % 3  

            ax = axes[row, col] if num_rows > 1 else axes[col] 
            
            if countplot:
                sns.countplot(data=data, x=feature, hue=hue, ax=ax)
            elif barplot:
                sns.barplot(data=data, x=feature, y=barplot_y, hue=hue, ax=ax)
            elif boxplot:
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax)
            elif outliers:
                sns.boxplot(data=data, x=feature, ax=ax)
            else:
                sns.histplot(data=data, x=feature, hue=hue, kde=kde, ax=ax)

            ax.set_title(feature)  
            ax.set_xlabel('')  
        
        if num_features < len(axes.flat):
            for j in range(num_features, len(axes.flat)):
                fig.delaxes(axes.flat[j])

        plt.tight_layout()
    
    except Exception as e:
        raise CustomException(e, sys)


def check_outliers(data, features):
    """
    Check for outliers in the given dataset features using the Interquartile Range (IQR) method.

    This function calculates the IQR and identifies outliers based on lower and upper bounds.
    It then provides statistics about the number and percentage of outliers for each feature.

    Args:
        data (pandas.DataFrame): The dataset containing the data to be analyzed.
        features (list): List of feature names for which outliers will be checked.

    Returns:
        tuple: A tuple containing three dictionaries:
               - Dictionary of outlier indexes for each feature.
               - Dictionary of outlier counts for each feature.
               - Total count of outliers in the dataset.

    Note:
        - The IQR method is used to identify outliers based on lower and upper bounds.
        - Outliers are detected if the feature values fall below the lower bound or above the upper bound.
        - The calculated IQR bounds use a factor of 1.5 times the IQR value.

    Example:
        outlier_indexes, outlier_counts, total_outliers = check_outliers(data=my_data, features=['Age', 'Income'])
    """

    try:
    
        outlier_counts = {}
        outlier_indexes = {}
        total_outliers = 0
        
        for feature in features:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            outlier_indexes[feature] = feature_outliers.index.tolist()
            outlier_count = len(feature_outliers)
            outlier_counts[feature] = outlier_count
            total_outliers += outlier_count
        
        print(f'There are {total_outliers} outliers in the dataset.')
        print()
        print(f'Number (percentage) of outliers per feature: ')
        print()
        for feature, count in outlier_counts.items():
            print(f'{feature}: {count} ({round(count/len(data)*100, 2)})%')

        return outlier_indexes, outlier_counts, total_outliers
    
    except Exception as e:
        raise CustomException(e, sys)