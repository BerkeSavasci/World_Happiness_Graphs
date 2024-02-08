import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

def create_and_save_histogram_by_group(file_name):
    xl = pd.ExcelFile(file_name)
    df = xl.parse('Tabellenblatt1')


    df['Ladder score'] = pd.to_numeric(df['Ladder score'], errors='coerce')
    df['Logged GDP per capita'] = pd.to_numeric(df['Logged GDP per capita'], errors='coerce')

    df = df.dropna(subset=['Ladder score', 'Logged GDP per capita'])

    median_threshold = df['Logged GDP per capita'].median()
    print(median_threshold)
    group_high_gdp = df[df['Logged GDP per capita'] > median_threshold]
    group_low_gdp = df[df['Logged GDP per capita'] <= median_threshold]

    sns.set_style('whitegrid')
    sns.set_palette('coolwarm')
    sns.histplot(group_high_gdp['Ladder score'], bins=30, kde=True)
    plt.xlabel('Ladder Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ladder Score in Europe (High GDP)')
    plt.savefig(f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/ladder_Score_T-Test_HighGDP{os.path.basename(file_name)}.png')
    plt.close()

    sns.set_style('whitegrid')
    sns.set_palette('coolwarm')
    sns.histplot(group_low_gdp['Ladder score'], bins=30, kde=True)
    plt.xlabel('Ladder Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ladder Score in Europe (Low GDP)')
    plt.savefig(f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/ladder_Score_T-Test_LowGDP{os.path.basename(file_name)}.png')
    plt.close()

    shapiro_test_high_gdp = stats.shapiro(group_high_gdp['Ladder score'])
    print(f'Shapiro test statistic (High GDP): {shapiro_test_high_gdp[0]}, p-value: {shapiro_test_high_gdp[1]}')

    shapiro_test_low_gdp = stats.shapiro(group_low_gdp['Ladder score'])
    print(f'Shapiro test statistic (Low GDP): {shapiro_test_low_gdp[0]}, p-value: {shapiro_test_low_gdp[1]}')

    t_statistic, p_value = stats.ttest_ind(group_high_gdp['Ladder score'], group_low_gdp['Ladder score'])
    print(f'T-test statistic: {t_statistic}, p-value: {p_value}')

def create_and_save_scatterplot(file_name):
    xl = pd.ExcelFile(file_name)
    df = xl.parse('Tabellenblatt1')

    df['Ladder score'] = pd.to_numeric(df['Ladder score'], errors='coerce')
    df['Logged GDP per capita'] = pd.to_numeric(df['Logged GDP per capita'], errors='coerce')
    df['Logged GDP per capita'] = df['Logged GDP per capita'].replace({',': '.'}, regex=True).astype(float)

    df = df.dropna(subset=['Ladder score', 'Logged GDP per capita'])

    sns.set_style('whitegrid')
    sns.set_palette('coolwarm')
    plt.xlabel('GDP')
    plt.ylabel('Ladder Score (Happiness degree)')
    plt.title('Correlation of GDP and Happiness Degree')
    sns.scatterplot(x='Logged GDP per capita', y='Ladder score', data=df)

    plt.savefig(
        f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/correlation_T-Test_{os.path.basename(file_name)}.png')
    plt.close()

def compute_correlation(file_name):
    xl = pd.ExcelFile(file_name)

    df = xl.parse('Tabellenblatt1')

    columns = [
        'Ladder score',
        'Logged GDP per capita',
        'Social support',
        'Healthy life expectancy',
        'Freedom to make life choices',
        'Generosity',
        'Perceptions of corruption'
    ]

    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.dropna(subset=columns)
    df = df.rename(columns={
        'Ladder score': 'Ladder Score',
        'Logged GDP per capita': 'GDP per Capita',
        'Social support': 'Social Support',
        'Healthy life expectancy': 'Life expectancy',
        'Freedom to make life choices': 'Freedom of Choice',
        'Generosity': 'Generosity',
        'Perceptions of corruption': 'Corruption'
    })
    new_columns = ['Ladder Score', 'GDP per Capita', 'Social Support',
                   'Life expectancy', 'Freedom of Choice', 'Generosity', 'Corruption']

    corr = df[new_columns].corr(method='spearman')

    plt.figure(figsize=(13, 13))

    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.savefig(
        f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/correlationTable_T-Test_{os.path.basename(file_name)}.png')
    plt.close()

file_list = [
    'Europe.xlsx',
]

for file_name in file_list:
    compute_correlation(file_name)
    create_and_save_histogram_by_group(file_name)
    create_and_save_scatterplot(file_name)
