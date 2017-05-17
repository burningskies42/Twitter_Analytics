from selenium import webdriver
from os import path

def get_chrome():
    chromedriver = (path.dirname(path.realpath(__file__))) + "\chromedriver.exe"
    # print(chromedriver)
    driver = webdriver.Chrome(chromedriver)
    return driver