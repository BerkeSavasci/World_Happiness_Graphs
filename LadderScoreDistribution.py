import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr


def create_and_save_histogram(file_name):
    # Load spreadsheet
    xl = pd.ExcelFile(file_name)
    # Parse the specified sheet into a DataFrame
    df = xl.parse('Tabellenblatt1')  # replace 'Tabellenblatt1' with your actual sheet name

    # Select the ladder score data
    ladder_scores = pd.to_numeric(df.iloc[:, 2].dropna(), errors='coerce')

    # Plotting histogram for ladder scores
    sns.set_style('whitegrid')
    sns.set_palette('coolwarm')
    sns.histplot(ladder_scores, bins=30, kde=True)
    plt.xlabel('Ladder Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ladder Score in Europe')
    plt.savefig(
        f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/ladder_Score{os.path.basename(file_name)}.png')
    plt.close()  # Close the plot after saving

    # Normality testing - Q-Q plot for ladder scores
    stats.probplot(ladder_scores, plot=plt)
    plt.title("QQ Diagramm - Ladder Scores")
    plt.savefig(
        f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/QQ_plot{os.path.basename(file_name)}_ladder_scores.png')
    plt.close()  # Close the plot after saving

    # Normality testing - Shapiro-Wilk test for ladder scores
    shapiro_test_ladder = stats.shapiro(ladder_scores)
    print(f'Ladder Scores - Shapiro test statistic: {shapiro_test_ladder[0]}, p-value: {shapiro_test_ladder[1]}')

    # Columns G, H, I, J
    for column in ['Ladder score', 'Logged GDP per capita', 'Social support', 'Perceptions of corruption',
                   'Freedom to make life choices']:
        # Select the data for the current column
        column_data = pd.to_numeric(df[column].dropna(), errors='coerce')

        # Plotting histogram
        sns.histplot(column_data, bins=30, kde=True)
        plt.xlabel(f'{column} Data')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {column} Data in Europe')
        plt.savefig(
            f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/histogram_{column.replace(" ", "_")}_{os.path.basename(file_name)}.png')
        plt.close()  # Close the plot after saving

        # Normality testing - Q-Q plot
        stats.probplot(column_data, plot=plt)
        plt.title(f"QQ Diagramm - {column} Data")
        plt.savefig(
            f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/QQ_plot_{column.replace(" ", "_")}_{os.path.basename(file_name)}.png')

        plt.close()  # Close the plot after saving

        # Normality testing - Shapiro-Wilk test
        shapiro_test_column = stats.shapiro(column_data)
        print(f'{column} Data - Shapiro test statistic: {shapiro_test_column[0]}, p-value: {shapiro_test_column[1]}')


def create_and_save_scatterplot(file_name):
    # Load spreadsheet
    xl = pd.ExcelFile(file_name)
    # Parse the specified sheet into a dataframe
    df = xl.parse('Tabellenblatt1')  # replace 'Tabellenblatt1' with your actual sheet name
    # Convert the ladder score (Happiness degree) and GDP columns to numeric
    df['Ladder score'] = pd.to_numeric(df['Ladder score'], errors='coerce')
    df['Logged GDP per capita'] = pd.to_numeric(df['Logged GDP per capita'], errors='coerce')
    df['Logged GDP per capita'] = df['Logged GDP per capita'].replace({',': '.'}, regex=True).astype(float)
    # Drop rows with missing values
    df = df.dropna(subset=['Ladder score', 'Logged GDP per capita'])
    # Scatter plot of GDP vs. Happiness degree
    sns.set_style('whitegrid')
    sns.set_palette('coolwarm')
    plt.xlabel('GDP')
    plt.ylabel('Ladder Score (Happiness degree)')
    plt.title('Correlation of GDP and Happiness Degree')
    sns.scatterplot(x='Logged GDP per capita', y='Ladder score', data=df)  # <-- This line was missing
    # Save the figure. Change '/path/to/your/directory/' with your actual file path.
    plt.savefig(
        f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/correlation_{os.path.basename(file_name)}.png')
    plt.close()  # Close the plot after saving


def compute_correlation(file_name):
    # Load spreadsheet
    xl = pd.ExcelFile(file_name)
    # Parse the specified sheet into a dataframe
    df = xl.parse('Tabellenblatt1')  # replace 'Tabellenblatt1' with your actual sheet name

    # List of columns we are interested in
    columns = [
        'Ladder score',
        'Logged GDP per capita',
        'Social support',
        'Healthy life expectancy',
        'Freedom to make life choices',
        'Generosity',
        'Perceptions of corruption'
    ]

    # Convert the ladder score (Happiness degree), GDP, and other columns to numeric
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows with missing values
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

    # Compute the correlation coefficients and p-values using Spearman correlation
    corr_matrix, p_values = spearmanr(df[new_columns], nan_policy='omit')

    plt.figure(figsize=(13, 13))

    # Visualize the correlation matrix with p-values
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Spearman correlation'})
    plt.title('Spearman Correlation Matrix with p-values')

    # Save the plot
    plt.savefig(
        f'/Users/berkesavasci/Documents/Datascience/World_Happiness_Graphs/graphs_LadderScore/correlationTable{os.path.basename(file_name)}.png')
    plt.close()

    # Display p-values (optional)
    gdp_ladder_p_value = p_values[new_columns.index('GDP per Capita'), new_columns.index('Ladder Score')]
    print(f"P-value for correlation between GDP per Capita and Ladder Score: {gdp_ladder_p_value}")


file_list = [
    'Europe.xlsx',
]
# Create and save a histogram for each file
for file_name in file_list:
    compute_correlation(file_name)
    create_and_save_histogram(file_name)
    create_and_save_scatterplot(file_name)
