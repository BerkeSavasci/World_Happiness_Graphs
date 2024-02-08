import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr

# fetch the data from excel
def load_data(file_name, sheet_name='Tabellenblatt1'):
    xl = pd.ExcelFile(file_name)
    return xl.parse(sheet_name)

# save plots with the given path
def save_plot(file_path):
    plt.savefig(file_path)
    plt.close()

# save the plot with specific name
def plot_histogram(data, column_name, file_name):
    sns.histplot(data, bins=30, kde=True)
    plt.xlabel(f'{column_name} Data')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column_name} Data in Europe')
    save_plot(f'plots/histogram_{column_name.replace(" ", "_")}_{os.path.basename(file_name)}.png')

# perform normality tests and create a QQ diagram
def normality_testing(data, column_name):
    stats.probplot(data, plot=plt)
    plt.title(f"QQ Diagramm - {column_name} Data")
    save_plot(f'plots/QQ_plot_{column_name.replace(" ", "_")}_{os.path.basename(file_name)}.png')

    shapiro_test_result = stats.shapiro(data)
    print(f'{column_name} Data - Shapiro test statistic: {shapiro_test_result[0]}, p-value: {shapiro_test_result[1]}')

# create and save histograms
def create_and_save_histograms(file_name, columns):
    df = load_data(file_name)

    for column in columns:
        column_data = pd.to_numeric(df[column].dropna(), errors='coerce')
        plot_histogram(column_data, column, file_name)
        normality_testing(column_data, column)

# scatter plot creation for visualization
def create_and_save_scatterplot(file_name):
    df = load_data(file_name)
    df['Ladder score'] = pd.to_numeric(df['Ladder score'], errors='coerce')
    df['Logged GDP per capita'] = pd.to_numeric(df['Logged GDP per capita'].replace({',': '.'}, regex=True),
                                                errors='coerce')
    df = df.dropna(subset=['Ladder score', 'Logged GDP per capita'])

    sns.scatterplot(x='Logged GDP per capita', y='Ladder score', data=df)
    plt.xlabel('GDP')
    plt.ylabel('Ladder Score (Happiness degree)')
    plt.title('Correlation of GDP and Happiness Degree')

    save_plot(f'plots/correlation_{os.path.basename(file_name)}.png')

# Compute and save Spearman correlation matrix for specified columns.
def compute_correlation(file_name):
    df = load_data(file_name)
    columns = ['Ladder score', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
               'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=columns)
    df = df.rename(columns={'Ladder score': 'Ladder Score', 'Logged GDP per capita': 'GDP per Capita',
                            'Social support': 'Social Support', 'Healthy life expectancy': 'Life expectancy',
                            'Freedom to make life choices': 'Freedom of Choice', 'Generosity': 'Generosity',
                            'Perceptions of corruption': 'Corruption'})

    new_columns = ['Ladder Score', 'GDP per Capita', 'Social Support', 'Life expectancy', 'Freedom of Choice',
                   'Generosity', 'Corruption']
    corr_matrix, p_values = spearmanr(df[new_columns], nan_policy='omit')

    plt.figure(figsize=(13, 13))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Spearman correlation'})
    plt.title('Spearman Correlation Matrix with p-values')

    save_plot(f'plots/correlationTable{os.path.basename(file_name)}.png')

    gdp_ladder_p_value = p_values[new_columns.index('GDP per Capita'), new_columns.index('Ladder Score')]
    print(f"P-value for correlation between GDP per Capita and Ladder Score: {gdp_ladder_p_value}")


file_list = ['tables/Europe.xlsx']

for file_name in file_list:
    compute_correlation(file_name)
    create_and_save_histograms(file_name,
                               ['Ladder score', 'Logged GDP per capita', 'Social support', 'Perceptions of corruption',
                                'Freedom to make life choices'])
    create_and_save_scatterplot(file_name)
