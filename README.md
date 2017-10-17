# Twitter Filter

This project contains several tools used for the purpose of my master thesis. The tools allow for mining data from Twitter, filtering it, labeling it and finally training classifiers

## Getting Started

Python 3.5 is required to run all the software in this repo
It is available at the official website: * [Python 3.5.0](https://www.python.org/downloads/release/python-350/) 
### Prerequisites
After installing the necessary Python release all the relevant modules can be installed with the help of Pip. Most of the required modules are denoted in the requirements.txt file. To install the models
you must open the command promt with admin privilges and navigate to the project folder. Once there run the command:


```
pip install -r requirements.txt
```

Additionally, several necessary packages are no longer available publicy from PyPi. Navigate to the [Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
4 packages must than installed:
[numpy-1.13.3+mkl-cp35-cp35m](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
[scipy-0.19.0-cp35-cp35m-(http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)
[PyQt4-4.11.4-cp35-cp35m](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4)
[statsmodels-0.8.0-cp35-cp35m](http://www.lfd.uci.edu/~gohlke/pythonlibs/#statsmodels)

Once downloaded, navigate to the appropriate folder and install packages with the following command:
```
pip install *.whl
```

### Scripts
