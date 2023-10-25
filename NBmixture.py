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

# follow these instructions
# https://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

import pandas as pd
import numpy as np
from scipy.stats import expon
from scipy.stats import norm
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix




class NBM_Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, gaussian_vars=None, multinom_vars=None, expon_vars=None, threshold = 0.5, adjust_zero_freq=False, preset_prevalence = None):
        self._more_tags()
        self.gaussian_vars = gaussian_vars
        self.multinom_vars = multinom_vars
        self.expon_vars = expon_vars
        self.threshold = threshold
        self.adjust_zero_freq = adjust_zero_freq
        self.preset_prevalence = preset_prevalence

        return

    def _more_tags(self):
        return {'allow_nan': True,}

    def fit(self, X, y):
        ### Check data types for X, y, and threshold
        assert (type(X) == np.ndarray), "X must be np.ndarray"
        assert ( (type(y) == np.ndarray) and (len(y) == len(X)) ), "y must be 1D np.ndarray where len(y) == len(X)"
        assert ( (type(self.threshold) == float) and (0.0 <= self.threshold < 1.0) ), "threshold must be float between 0.0 and 1.0"

        ### default all gaussian variables if nothing provided 
        if (not self.gaussian_vars) and (not self.multinom_vars) and (not self.expon_vars):
            self.gaussian_vars = list(np.arange(X.shape[1]))

        # test all other parameters
        if self.preset_prevalence:
            assert (type(self.preset_prevalence) == dict), " must be dictionar of format: {'0.0': 0.5, '1.0': 0.5}. Outcome values are dictionary keys)"
        if self.gaussian_vars:
            assert (type(self.gaussian_vars) == list), "gaussian_vars parameter must be list"
        if self.multinom_vars:
            assert (type(self.multinom_vars) == list), "multinom_vars parameter must be list"
        if self.expon_vars:
            assert (type(self.expon_vars) == list), "expon_vars parameter must be list"

        # set the levels of the outcome variable
        self.classes_ = self.get_levels(y)

        ### Train the Classifier
        # compute the prevaluence of each outcome class
        if not self.preset_prevalence:
            self.prevalence_params_ = self.train_prevalence(X, y)
        else:
            self.prevalence_params_ = self.preset_prevalence

        # compute the Conditional Linear Gaussian parameters for Gaussian variables
        if self.gaussian_vars:
            self.CLG_list_ = self.train_CLG_variables(X, y)

        # compute the Tabular CPD parameters for our binomial variables
        if self.multinom_vars:
            self.CPT_list_ = self.train_multinomial_variables(X, y)

        # compute the Conditional Linear Gaussian parameters for exponential variables
        if self.expon_vars:
            self.Expon_list_ = self.train_Exponential_variables(X, y)

        return self

    @staticmethod
    def get_levels(v):
        levels = np.unique(v)
        levels = levels[~np.isnan(levels)]
        return levels

    def train_prevalence(self, X, y):

        prevalence_params = {}

        for l_y in self.classes_:
            prev = sum(y == l_y) / float(len(y))
            prevalence_params.update({l_y:prev})

        return prevalence_params
    
    def train_CLG_variables(self, X, y):

        CLG_list = {}
        for var in self.gaussian_vars:
            CLG = self.generate_CLG(X[:,var], y)

            CLG_list.update({var:CLG})

        return CLG_list
    
    def train_multinomial_variables(self, X, y):

        CPT_list = {}
        for var in self.multinom_vars:
            try:
                CPT = self.generate_CPT(X[:,var], y)
            except ZeroDivisionError:
                print('ZeroDivisionError: there is no data entered for the variable %s' %(var))

            CPT_list.update({var:CPT})

        return CPT_list

    def train_Exponential_variables(self, X, y):

        Expon_list = {}
        for var in self.expon_vars:
            Exponential = self.generate_Expon(X[:,var], y)

            Expon_list.update({var:Exponential})

        return Expon_list

    def predict(self, X, y=None):
        ### Check data types for X, y, and threshold
        assert (type(X) == np.ndarray), "X must be np.ndarray"
        if y is not None:
            assert( (type(y) == np.ndarray) or (len(y) == len(X)) ), "y must be 1D np.ndarray where len(y) == len(X)"

        # compute Probability of each Class in outcome_var
        log_class_probabilities = {}
        for class_i in self.classes_:
            # class_i = str(float(class_i))

            # generate aray of conditional probabilitis for each feature 
            conditional_probs = []
            # for var in input_data.columns:
            # compute for Gaussian variables based on Conditional Linear Gaussian 
            for var in self.gaussian_vars:
                p = np.vectorize(self.CLG_calculateProbability, excluded=['CLG'])(X[:,var], self.CLG_list_[var], class_i)
                conditional_probs.append(p)

            # compute for multinomial variables based on Tabular Conditional Probability Distribution 
            for var in self.multinom_vars:
                p = np.vectorize(self.CPT_calculateProbability, excluded=['CPT'])(X[:,var], self.CPT_list_[var], class_i)
                conditional_probs.append(p)

            # compute for exponential variables based on Conditional Linear Gaussian 
            for var in self.expon_vars:
                p = np.vectorize(self.Expon_calculateProbability, excluded=['Expon'])(X[:,var], self.Expon_list_[var], class_i)
                conditional_probs.append(p)

            # convert to np.array
            conditional_probs = np.array(conditional_probs)

            # temporarily replace nan's with 1.0
            conditional_probs[np.isnan(conditional_probs)]=1.0

            if self.adjust_zero_freq == False:
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
            log_prior_p = np.log(self.prevalence_params_[class_i])
            log_cls_probs = log_cls_probs + log_prior_p
            
            # add to dictionary with result for each class
            log_class_probabilities.update({class_i:log_cls_probs})

        self.log_class_probabilities_ = log_class_probabilities
        # self.classes_in_order_ = list(self.log_class_probabilities_.keys())

        # compute the predicted class
        pred_proba_ = np.array([v for k,v in self.log_class_probabilities_.items()])
        pred_proba_ = pred_proba_.T

        self.pred_proba_ = pred_proba_
        self.prediction_ = self.classes_[np.argmax(pred_proba_, axis=1)]

        # self.log_probabilities_ = []
        # for i,a in enumerate(argmax):
        #     self.log_probabilities_.append(np.exp(pred_proba_[a,i]))

        return self.prediction_

    def predict_proba(self, X, y):

        _prediction = self.predict(X, y)

        return self.pred_proba_

    def score(self, X, y, verbose=False):
        
        _prediction = self.predict(X, y)

        outcome = []
        for i in range(len((self.prediction_))):
            pred_i = self.prediction_[i]

            if (pred_i == 'nan') or (y[i] == 'nan'):
                score_i = np.nan
            else:
                score_i = pred_i == y[i]

            outcome.append(score_i)

        outcome = np.array(outcome)

        if verbose == False:
            accuracy = np.nanmean(outcome)

        else:
            # compute metrics
            # remove nans for computing metrics
            y_p_pairs = np.array([[i,j] for i,j in zip(y,_prediction) if (np.isnan(i)==False) and (np.isnan(j)==False)]).T

            y_no_nan = y_p_pairs[0]
            pred_no_nan = y_p_pairs[1]

            # confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_no_nan, pred_no_nan).ravel()
            matrix_total = tn + fp + fn + tp
            accuracy = (tp+tn)/matrix_total
            P_correct = (tp+fp)/matrix_total * (tp+fn)/matrix_total
            P_incorrect = (fn+tn)/matrix_total * (fp+tn)/matrix_total
            P_e = P_correct+P_incorrect

            self.metrics_ = {'accuracy': accuracy,
                             'sensitivity': tp/(tp+fn),
                             'specificity': tn/(tn+fp),
                             'PPV': tp/(tp+fp),
                             'NPV': tn/(tn+fn),
                             'F1': tp/(tp+0.5*(fp+fn)),
                             'Cohens_Kappa': (accuracy - P_e)/(1.0 - P_e)}

            print('algorithm performance:')
            print(self.metrics_)
        
        self.outcome_ = outcome
        self.accuracy_ = accuracy
        
        return self.accuracy_

    def generate_CLG(self, x, y):
        CLG_params = {}
        for l_y in self.classes_:
            d_l = np.array([i for i,j in zip(x,y) if j == l_y])
            d_l = d_l[~np.isnan(d_l)]

            CLG_params.update({l_y: [d_l.mean(),d_l.std()]})

        return CLG_params

    def generate_CPT(self, x, y):

        x_levels = self.get_levels(x)


        counts = np.zeros([len(self.classes_), len(x_levels)])

        for i in range(len(x)):

            if np.isnan(x[i]) or np.isnan(y[i]):
                continue

            counts[np.where(self.classes_ == y[i])[0][0], np.where(x_levels == x[i])[0][0]] += 1

        # handle the "zero frequency problem" for instances returning P(y|x)=0
        if self.adjust_zero_freq == True:
            if 0 in counts:
                counts += 1

        # turn counts into conditional probabilities
        counts = counts/counts.sum(axis=1, keepdims=True) 

        CPT = dict()

        for i in range(len(self.classes_)):
            factor_probs = dict()
            
            for j in range(len(x_levels)):
                factor_probs.update({x_levels[j]:counts[i,j]})
        
            CPT.update({self.classes_[i]:factor_probs})

        return CPT

    def generate_Expon(self, x, y):
        Expon_params = {}
        for l_y in self.classes_:
            d_l = np.array([i for i,j in zip(x,y) if j == l_y])
            d_l = d_l[~np.isnan(d_l)]

            # fit exponential distribution
            Expon_position_beta = expon.fit(d_l)
            
            # add to dictionary
            Expon_params.update({l_y: list(Expon_position_beta)})

        return Expon_params

    @staticmethod
    def CLG_calculateProbability(x, CLG, class_i):
        '''
        Here is our function for calculating the probability that an unknown entry belongs to a certain class
        for our Conditional Linear Gaussian
        '''
        if np.isnan(x):
            return np.nan

        else:
            x = float(x)
            # class_i = str(float(class_i))

            mean = CLG[class_i][0]
            stdev = CLG[class_i][1]

            p = 2.0 * norm.cdf(-1.0 * np.abs(x-mean), 0,stdev) ## <-Cumulative Probability Distribution * 2

            return p

    @staticmethod
    def CPT_calculateProbability(x, CPT, class_i):
        '''
        Here is our function for calculating the probability that an unknown entry belongs to a certain class
        for our Conditional Probability Distributions
        '''

        if np.isnan(x):
            return np.nan

        else:
            x = float(x)
            # class_i = str(float(class_i))

            try:
                p = CPT[class_i][x]

            except KeyError:
                print('KeyError: returning NAN')
                p = np.nan

            return p

    @staticmethod
    def Expon_calculateProbability(x, Expon, class_i):
        '''
        Calculate the probability that an unknown entry belongs to a certain class
        for our exponential distributions
        '''

        if np.isnan(x):
            return np.nan

        else:
            x = float(x)
            # class_i = str(float(class_i))

            position = Expon[class_i][0]
            beta = Expon[class_i][1]

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

