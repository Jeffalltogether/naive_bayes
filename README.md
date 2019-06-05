# Naive Bayes Classifier  
This is a Python implementation of a Naive Bayes Classifier that has two useful attributes:  
  
1. continuous features can be merged with multinomial features; and  
2. missing data in both the output class and input features will be handled without error.  
  
Please see the file `Example_Naive_Bayes.ipynb` for a walkthrough.  
  
Continuous variables are modeled with a conditional linear Gaussian, while multinomial variables are modeled with a tabular conditional probability distribution. 
  
## Dependencies   
The classifier `Naive_Bayes_Model.py` was written in the following environemnt:  
* Python 2.7.16  
* NumPy 1.14.3  
* Pandas 0.24.2  
  
  
## Please Contribute  
If you find an error, better way to compute the parameters of the algorithm, or have a new feature to add please submit a pull request.  
