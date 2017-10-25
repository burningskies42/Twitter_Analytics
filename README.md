
Twitter Filter
-------------

This project contains several tools used for the purpose of my master thesis. The tools allow for mining data from Twitter, filtering it, labeling it and finally training classifiers

> **Note:**

> - Twitter Filter is a loose collection of tools designed for minining Twitter data, rather then a complete software package
> - All software is based on Python 3.5.0


## Getting Started

Python 3.5 is required to run all the software in this repo
It is available at the official website: [Python 3.5.0](https://www.python.org/downloads/release/python-350/) 
### Prerequisites
After installing the necessary Python release all the relevant modules can be installed with the help of Pip. Most of the required modules are denoted in the requirements.txt file. To install the models
you must open the command promt with admin privilges and navigate to the project folder. Once there run the command:
```
pip install -r requirements.txt
```

Additionally, several necessary packages are no longer available publicy from PyPi. Navigate to the [Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
4 packages must than installed:
>- [numpy-1.13.3+mkl-cp35-cp35m](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
>- [scipy-0.19.0-cp35-cp35m](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)
>- [PyQt4-4.11.4-cp35-cp35m](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4)
>- [statsmodels-0.8.0-cp35-cp35m](http://www.lfd.uci.edu/~gohlke/pythonlibs/#statsmodels)

Once downloaded, navigate to the appropriate folder and install packages with the following command (replace * with appropriate file name):
```
pip install *.whl
```

### Scripts
 **View_bot_suspects** 
 *Displays a list of Twitter users ids, suspected in being bots *
 
 **word_frequency** 
*Creates a frequency list of all words in a given Twitter corpus*

 **..._features_test** 
 *Runs a simulation of training and testing differenct classifiers using the appropriate method \*. Possible feature types: Descriptive, Bag-of-Words, N-Grams *
 
 