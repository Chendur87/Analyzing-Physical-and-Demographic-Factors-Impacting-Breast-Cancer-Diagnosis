import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import random


class ResearchQuestionOne:
    '''
    Runs the functions pertaining to my first research question:
    To what extent do the features of tumors (size, uniformity, etc.)
    correlate to tumors being either malignant (cancerous) or benign
    (non-cancerous)?
    '''

    def __init__(self):
        '''
        Sets up the datasets and samples needed to answer the research question.
        '''
        self.df = pd.read_csv(
            'datasets/data.csv').loc[:, 'diagnosis':'symmetry_mean']

        # separating the sample into cancer & non-cancer patients
        self.df_malignant = self.df[self.df['diagnosis'] == 'M']
        self.df_benign = self.df[self.df['diagnosis'] == 'B']

        # taking a random sample from each population for the
        # confidence interval (ci) & hypothesis testing results
        self.malignant_sample = self.df_malignant.sample(
            n=15, random_state=random.randint(0, 1000))
        self.benign_sample = self.df_benign.sample(n=15,
                                                   random_state=random.randint(
                                                       0, 1000))
        # to store confidence interval (ci) & hypothesis testing results
        self.df_statistics = pd.DataFrame()

    def run(self):
        '''
        Runs through all methods needed to get the info to answer the research
        question.
        '''
        self.plot_distributions(self.df, 'research_question_one_plots/')
        self.calculate_ci()
        # printing the statistics on the console for easy viewing
        print(
            self.conduct_hypo_testing(self.malignant_sample,
                                      self.benign_sample)[2])

    def plot_distributions(self, df, directory):
        """
        Plots the distributions of cancer & non-cancer patients of the following
        features of the tumors: radius, texture, perimeter, area, smoothness,
        compactness, concavity, # of concave points, and symmetry
        """
        # plot the distributions for each type of mean
        # with the hue as the diagnosis to show the
        # potential differences between cancerous vs
        # non-cancerous in these measurements
        for col in df.columns[1:]:
            sns.histplot(data=df, x=col, hue='diagnosis')
            plt.savefig(directory + col + '.png')
            plt.clf()

    def calculate_ci(self):
        """
        Calculates the confidence intervals of each of the features
        for the both the cancer and non-cancer patient populations
        """
        for col in self.malignant_sample.loc[:, 'radius_mean':'symmetry_mean']:

            # get the means and standard deviations of both samples
            # (cancer & non-cancer patients)
            malignant_mean = self.malignant_sample[col].mean()
            benign_mean = self.benign_sample[col].mean()
            malignant_std = self.malignant_sample[col].std()
            benign_std = self.benign_sample[col].std()

            # to calculate a confidence interval of differences:
            # must calcuate a population estimate of the differences
            # of the means between the two populations
            mean_diff = malignant_mean - benign_mean

            # standard_error calculation:
            # sqrt(std(1)^2 / n(1) + std(2)^2 / n(2))
            # where std(1) and n(1) are for one sample
            # and std(2) and n(2) are for the other
            # std is standard deviation and n is sample size
            standard_error = (
                (malignant_std)**2 / len(self.malignant_sample[col]) +
                (benign_std)**2 / len(self.benign_sample[col]))**0.5

            # the critical t value is a value that determines how large the
            # confidence interval based on the confidence level
            # Because this is a confidence interval for means, we must
            # conduct a t-interval for a difference of means instead of
            # finding a critical z value for proportions
            critical_t = st.t.ppf(
                0.975,
                len(self.malignant_sample[col]) +
                len(self.benign_sample[col]) - 2)

            #  2 sample confidence-interval formula:
            #  mean diff +/- critical_t + standard error
            self.df_statistics[
                col] = mean_diff - critical_t * standard_error, \
                        mean_diff + critical_t * standard_error

        self.df_statistics = self.df_statistics.T
        self.df_statistics = self.df_statistics.rename(columns={
            0: 'lower_bound_ci',
            1: 'upper_bound_ci'
        })

    def conduct_hypo_testing(self, malignant_sample, benign_sample):
        """
        Conduct hypothesis testing to see whether there is a statistically
        significant difference in each feature (radius_mean to symmetry_mean)
        on the graph

        Null hypothesis for each feature: The true difference between the 
        populations with and without breast cancer in the measurement of the
        feature is 0 (minimal so that it's statistically INsignificant).

        Alternative hypothesis for each feature: The true difference between  
        the populations with and without breast cancer in the measurement of 
        the feature is NOT 0 (minimal so that it's statistically SIGNIFICANT)
        """

        t_statistics = []
        p_values = []

        # gather the t-statistic and p-value for each feature that
        # was considered for the confidence interval, leveraging
        # the scipy.stats module to conduce a 2-sided test (alternative
        # hypotheis is NOT 0, which is either less than or greater than
        # 0, thus two-sided)
        for col in malignant_sample.loc[:, 'radius_mean':'symmetry_mean']:
            t_statistic, p_value = st.ttest_ind(malignant_sample[col],
                                                benign_sample[col],
                                                alternative='two-sided')
            t_statistics.append(t_statistic)
            p_values.append(float('{:.6f}'.format(p_value)))
        self.df_statistics['p_value'] = p_values

        # if based on the p_values < 0.05, there IS a statistically
        # significant difference between the two populations (rejecting
        # the null hypothesis), else there isn't a significant difference
        # (failing to reject the null)
        self.df_statistics[
            'Statistically Significant Difference'] = self.df_statistics[
                'p_value'] < 0.05

        # will utilize these values in testing
        return t_statistics, p_values, self.df_statistics
