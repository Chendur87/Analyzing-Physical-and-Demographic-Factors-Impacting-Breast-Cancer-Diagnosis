import pandas as pd
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


class ResearchQuestionTwo:
    """
    Runs the functions pertaining to my first research question:
    How strong is the correlation between the demographic factors
    (age, race, marital status) of the patient and the features of
    the tumors (size, grade of tumor, percent carcinogenic)?
    How big of a role do factors outside of the tumor area correlate
    to factors within the tumor area?
    """

    def __init__(self):
        '''
        Sets up the dataset needed to answer the research question.
        '''
        self.df_demographics = pd.read_csv(
            'datasets/Breast_Cancer.csv').rename(
                columns={
                    'T Stage ': 'T Stage',
                    'Reginol Node Positive': 'Regional Node Positive',
                })
        # a column that was more descriptive than just number of nodes looked
        # at or how many carcinogenic nodes were there
        self.df_demographics[
            'Proportion of Positive Regional Nodes'] = self.df_demographics[
                'Regional Node Positive'] / self.df_demographics[
                    'Regional Node Examined']
        self.directory = 'research_question_two_plots/'

    def run(self):
        '''
        Runs through plotting both the categorical & regression plots
        needed to answer the question.
        '''

        self.plot_categorical()
        self.plot_regression()

    def plot_categorical(self):
        """
        Plots race & marital status (categorical demographic variables)
        in comparison to features of the breast tissue.
        """
        # tumor size vs race & marital status
        for x_var in ['Race', 'Marital Status']:
            sns.barplot(data=self.df_demographics,
                        x=x_var,
                        y='Tumor Size',
                        hue="T Stage")
            # .lower.split() for consistent naming convention on plots
            plt.savefig(self.directory +
                        f'tumor_size_vs_{"_".join(x_var.lower().split())}.png')
            plt.clf()

        # age vs grade of the tumor
        sns.barplot(data=self.df_demographics, x='Grade', y='Age')
        plt.savefig(self.directory + 'age_vs_grade.png')
        plt.clf()

        # proportion of node examined that were positive vs
        # race & marital status
        for x_var in ['Race', 'Marital Status']:
            sns.barplot(data=self.df_demographics,
                        x=x_var,
                        y='Proportion of Positive Regional Nodes',
                        hue="T Stage")
            plt.savefig(
                self.directory +
                f'prop_+_nodes_vs_{"_".join(x_var.lower().split())}.png')
            plt.clf()

        # number of survival months vs race & marital status
        for x_var in ['Race', 'Marital Status']:
            sns.barplot(data=self.df_demographics,
                        x=x_var,
                        y='Survival Months')
            plt.savefig(
                self.directory +
                f'survival_months_vs_{"_".join(x_var.lower().split())}.png')
            plt.clf()

    def plot_regression(self):
        """
        Plots age (quantitative demographic variable) with other quantitative
        features to create regression plots
        """
        # filter to just quantitative columns that uses the 'Age' column to
        # groupby to only plot the means
        df = self.df_demographics[[
            'Age', 'Tumor Size', 'Survival Months',
            'Proportion of Positive Regional Nodes', 'Regional Node Examined'
        ]].groupby('Age').mean()

        # lists are for testing
        r_values = []
        reg_plots = []
        slopes = []

        # plot each of these output variables in comparison to age
        for y_var in [
                'Proportion of Positive Regional Nodes', 'Tumor Size',
                'Regional Node Examined', 'Survival Months'
        ]:

            plot = sns.regplot(data=df, x=df.index, y=y_var)
            # this is the r coefficient that determines the strength of the
            # association between the two variables
            r = st.pearsonr(df.index, df[y_var])[0]
            # statistics for testing
            r_values.append(r)
            reg_plots.append(plot)
            b = st.linregress(x=plot.get_lines()[0].get_xdata(),
                              y=plot.get_lines()[0].get_ydata())[0]
            slopes.append(b)
            # adding the r coefficient to the plots
            plt.text(x=0.90,
                     y=0.95,
                     s="r: {:.2f}".format(r),
                     ha='center',
                     va='top',
                     transform=plt.gca().transAxes)
            name = '_'.join(y_var.lower().split())
            plt.savefig(self.directory + f'{name}_vs_age.png')
            plt.clf()

        # return values for testing purposes
        return df, r_values, reg_plots, slopes
