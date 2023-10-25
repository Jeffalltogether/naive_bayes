# MIT License

# Copyright (c) 2019 Jeffrey Thatcher

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pandas as pd
import numpy as np
# from pandas.core.base import DataError
from scipy.stats import expon
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator




class NB_Classifier(BaseEstimator, ClassifierMixin):
    import numpy as np
    import pandas as pd

    def __init__(self, training_df=[], outcome_variable = 'y', gaussian_variables = [], multinomial_variables = [], exponential_variables = []):
        self.data = training_df
        self.outcome_var = outcome_variable
        self.gaussian_vars = gaussian_variables
        self.nom_vars = multinomial_variables
        self.expon_vars = exponential_variables

        return

    def fit(self):
        # compute the prevaluence of each outcome class
        self.train_prevalence()

        # compute the Conditional Linear Gaussian parameters for Gaussian variables
        self.train_CLG_variables()

        # compute the Conditional Linear Gaussian parameters for exponential variables
        self.train_Exponential_variables()

        # compute the Tabular CPD parameters for our binomial variables
        self.train_multinomial_variables()

        return
        
        
    def train_prevalence(self):
        sub_df = self.data[self.outcome_var].dropna()

        y_levels = np.unique(sub_df.values)
        y_levels = y_levels[~np.isnan(y_levels)]

        prevalence_params = {}
        for l_y in y_levels:
            prev = sum(sub_df.values == l_y) / float(len(sub_df))
            prevalence_params.update({str(float(l_y)):prev})

        self.prevalence_params = prevalence_params
        
        return
    
    def train_CLG_variables(self):

        CLG_list = {}
        for var in self.gaussian_vars:
            try:
                CLG = self.generate_CLG(self.data, self.outcome_var, var)
            except:
            	print('DataError: data in the variable %s is not numeric, please convert to numeric and re-run' %(var))

            CLG_list.update({str(var):CLG})

        self.CLG_list = CLG_list
        
        return
    
    def train_multinomial_variables(self):

        CPD_list = {}
        for var in self.nom_vars:
            try:
                CPD = self.generate_CPD(self.data, self.outcome_var, var)
            except ZeroDivisionError:
                print('ZeroDivisionError: there is no data entered for the variable %s' %(var))

            CPD_list.update({str(var):CPD})

        self.CPD_list = CPD_list
        
        return

    def train_Exponential_variables(self):

        Expon_list = {}
        for var in self.expon_vars:
            try:
                Exponential = self.generate_Expon(self.data, self.outcome_var, var)
            except:
            	print('DataError: data in the variable %s is not numeric, please convert to numeric and re-run' %(var))

            Expon_list.update({str(var):Exponential})

        self.Expon_list = Expon_list
        
        return

    def predict(self, test_data = pd.DataFrame([]), hyper_prevalence_params = None):
        # grab test data if given
        if not test_data.empty:
            print('Inferencing on Test Data')
            self.data = test_data
        else:
            print('Inferencing on Training Data')
        
        # drop the outcome variable from the input data frame
        input_data = self.data.drop(self.outcome_var, axis=1)

        # get the levels of the outcome variable excluding the nan's
        y_levels = np.unique(self.data[self.outcome_var])
        y_levels = y_levels[~np.isnan(y_levels)]

        # compute Probability of each Class in outcome_var
        log_class_probabilities = {}
        for class_i in y_levels:
            class_i = str(float(class_i))

            # generate aray of conditional probabilitis for each feature 
            conditional_probs = []
            for var in input_data.columns:
                # compute for Gaussian variables based on Conditional Linear Gaussian 
                if var in self.gaussian_vars:
                    p = np.vectorize(self.CLG_calculateProbability, excluded=['CPD'])(input_data[var], self.CLG_list[var], class_i)
                    conditional_probs.append(p)

                # compute for multinomial variables based on Tabular Conditional Probability Distribution 
                if var in self.nom_vars:
                    p = np.vectorize(self.CPT_calculateProbability, excluded=['CPD'])(input_data[var], self.CPD_list[var], class_i)
                    conditional_probs.append(p)

                # compute for exponential variables based on Conditional Linear Gaussian 
                if var in self.expon_vars:
                    p = np.vectorize(self.Expon_calculateProbability, excluded=['CPD'])(input_data[var], self.Expon_list[var], class_i)
                    conditional_probs.append(p)

            # convert to np.array
            conditional_probs = np.array(conditional_probs)


            # temporarily replace nan's with 1.0
            conditional_probs[np.isnan(conditional_probs)]=1.0

            # replace 0.0 with very small number so that we can log transform without gettin -inf
            conditional_probs = np.where(conditional_probs==0, 1.0e-12, conditional_probs)

            # multiply conditional proability for each feature from each observation 
            # Note: use log probabilities to avoid difficulty with the precision of floating point values
            # math: log(a*b) = log(a) + log(b)
            
            # get log(p) of first CPs
            log_cls_probs = np.zeros(len(conditional_probs[0]))

            # Add remining log(p)'s
            for i in range(len(conditional_probs)):
                log_cls_probs += np.log(conditional_probs[i])

            # if any are exactly 0.0 we know that all entries for that observation were nan's, we can replace these with nan
            # log_cls_probs[log_cls_probs==0.0]=np.nan

            # multiply the resuling probabilities with the prevalence
            if hyper_prevalence_params == None:
                log_prior_p = np.log(self.prevalence_params[str(float(class_i))])
                log_cls_probs = log_cls_probs + log_prior_p
            else:
                print('Using user-defined prevalence parameters')
                log_prior_p = np.log(hyper_prevalence_params[str(float(class_i))])
                log_cls_probs = log_cls_probs + log_prior_p

            # add to dictionary with result for each class
            log_class_probabilities.update({class_i:log_cls_probs})

        self.log_class_probabilities = log_class_probabilities
        self.classes_in_order = list(self.log_class_probabilities.keys())

        # compute the predicted class
        pred_prob_results = np.array([v for k,v in self.log_class_probabilities.items()])
        argmax = np.array([i for i in np.argmax(pred_prob_results, axis=0)])

        self.prediction = [self.classes_in_order[i] for i in argmax]

        self.log_probabilities = []
        for i,a in enumerate(argmax):
            self.log_probabilities.append(np.exp(pred_prob_results[a,i]))

        return

    def compute_results(self):
        # compute the actual class
        ground_truth = np.array([str(float(i)) for i in self.data[self.outcome_var].values])

        outcome = []
        for i in range(len((self.prediction))):

            if self.prediction[i] == 'nan':
                comp = np.nan
            elif ground_truth[i] == 'nan':
                comp = np.nan
            else:
                comp = self.prediction[i] == ground_truth[i]

            outcome.append(comp)

        # compute accuracy
        outcome_array = np.array(outcome)

        outcome_array = outcome_array[~np.isnan(outcome_array)]    

        accuracy = float(sum(outcome_array)) / len(outcome_array)
        print('algorithm accuracy: %.5f' %(accuracy))

        self.ground_truth = ground_truth
        self.outcome = outcome
        self.accuracy = accuracy
        
        return

    @staticmethod
    def generate_CLG(data, outcome_var = 'y', cont_var = 'x'):
        # separate the data
        sub_df = data[[outcome_var, cont_var]]

        # compute levels of outcome variable
        y_levels = np.unique(sub_df[outcome_var])
        y_levels = y_levels[~np.isnan(y_levels)]

        # compute the means and standard deviations for each continuous variable
        CLG_means = sub_df.groupby([outcome_var], as_index=False).mean()
        CLG_stds = sub_df.groupby([outcome_var], as_index=False).std()

        CLG_params = {}
        for l_y in y_levels:
            CLG_params.update({str(float(l_y)): []})

        for row_m, row_s in zip(CLG_means.values, CLG_stds.values):
            CLG_params.update({str(row_m[0]): [row_m[1], row_s[1]]})

        return CLG_params

    @staticmethod
    def generate_CPD(data, outcome_var = 'y', nom_var = 'x'):

        df2 = data[[outcome_var,nom_var]].reset_index().groupby([outcome_var, nom_var]).count().unstack(nom_var).fillna(0)
        df2.columns = df2.columns.droplevel(0)

        # # handle the "zero frequency problem" for instances returning P(y|x)=0
        # if 0 in df2.values:
        #     df2 += 1 # add 1 to every cell in the frequency counts

        df3 = df2.divide(df2.sum(axis=1), axis=0)

        CPD = dict()
        for row in df3.iterrows():
            i = row[0]
            CPD.update({ str(float(row[1].index[i])): { str(float(val)):prob for val,prob in zip(row[1].index,row[1]) } })

        return CPD

    @staticmethod
    def generate_Expon(data, outcome_var = 'y', cont_var = 'x'):
        # separate the data
        sub_df = data[[outcome_var, cont_var]]

        # compute levels of outcome variable
        y_levels = np.unique(sub_df[outcome_var])
        y_levels = y_levels[~np.isnan(y_levels)]

        # compute the start position and beta for each exponential variable
        Expon_params = {}
        for level in y_levels:
            # get x data from our small pandas dataframe
            x = sub_df.loc[sub_df[outcome_var] == level]
            x = x[cont_var].values
            x = x[~np.isnan(x)]

            # fit exponential distribution
            Expon_position_beta = expon.fit(x)
            
            # add to dictionary
            Expon_params.update({str(float(level)): list(Expon_position_beta)})

        return Expon_params

    @staticmethod
    def CLG_calculateProbability(x, CPD, class_i):
        '''
        Here is our function for calculating the probability that an unknown entry belongs to a certain class
        for our Conditional Linear Gaussian
        '''
        # Calculate the Gaussian probability distribution function for x

        def calculate_probability(xx, mm, ss):
            exponent = np.exp(-( (xx-mm)**2 / (2 * ss**2 ) ) ) 
            return (1 / (np.sqrt(2 * 3.141592653589793) * ss)) * exponent



        import scipy.stats

        if np.isnan(x):
            return np.nan

        else:
            x = float(x)
            class_i = str(float(class_i))

            mean = CPD[class_i][0]
            stdev = CPD[class_i][1]

            p = 2.0 * scipy.stats.norm.cdf(-1.0 * np.abs(x-mean), 0,stdev) ## <-Cumulative Probability Distribution * 2

            return p

    @staticmethod
    def CPT_calculateProbability(x, CPD, class_i):
        '''
        Here is our function for calculating the probability that an unknown entry belongs to a certain class
        for our Conditional Probability Distributions
        '''
        if np.isnan(x):
            return np.nan

        else:
            x = str(float(x))
            class_i = str(float(class_i))

            try:
                p = CPD[class_i][x]

            except KeyError:

                print('KeyError: returning NAN')
                p = np.nan

            return p

    @staticmethod
    def Expon_calculateProbability(x, CPD, class_i):
        '''
        Calculate the probability that an unknown entry belongs to a certain class
        for our exponential distributions
        '''

        if np.isnan(x):
            return np.nan

        else:
            x = float(x)
            class_i = str(float(class_i))

            position = CPD[class_i][0]
            beta = CPD[class_i][1]

            fit_exponential = expon(position, beta)
            p = 1-fit_exponential.cdf(x)

            return p


if __name__ == "__main__":


	y = np.concatenate((np.zeros(10),np.ones(10)), axis=0)

	x1 = np.random.normal(loc=1, scale=0.2, size=20)
	x2 = y + np.random.normal(loc=0.0, scale=0.2, size=20)
	x3 = np.concatenate((np.random.binomial(n=1, p=0.85, size = 10), np.random.binomial(n=1, p=0.15, size = 10)), axis = 0)
	x4 = np.random.binomial(n=1, p=0.5, size = 20)

	train_df = pd.DataFrame()
	train_df['y'] = y
	train_df['x1'] = x1
	train_df['x2'] = x2
	train_df['x3'] = x3
	train_df['x4'] = x4
	train_df



	training_df = train_df
	Gaussian_variables = ['x1','x2']
	multinomial_variables = ['x3','x4']
	outcome_variable = 'y'

	my_NB = NB_Classifier(training_df, outcome_variable, Gaussian_variables, multinomial_variables)
	my_NB.train_prevalence()
	my_NB.train_CLG_variables()
	my_NB.train_multinomial_variables()
	my_NB.compute_conditional_probs()
	my_NB.compute_results()



	y = np.concatenate((np.zeros(10),np.ones(10)), axis=0)
	x1 = np.zeros(20)
	x2 = np.zeros(20)
	x3 = np.zeros(20)
	x4 = np.zeros(20)

	test_df = pd.DataFrame()
	test_df['y'] = y
	test_df['x1'] = x1
	test_df['x2'] = x2
	test_df['x3'] = x3
	test_df['x4'] = x4
	test_df

	my_NB.compute_conditional_probs(test_df)
	my_NB.compute_results()

