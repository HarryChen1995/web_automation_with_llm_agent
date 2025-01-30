from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from langchain_community.document_loaders import WebBaseLoader
import time





chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=chrome_options)


def google_search(driver, query:str):
    driver.get(f"https://www.google.com/search?q={query}")
    time.sleep(7)
    cites = driver.find_elements(By.XPATH, '//cite')
    urls = list(map(lambda x:x.text.split(" › ")[0], filter(lambda x:  x.text, cites)))
    return urls
    
    

def go_to_web_page(driver, link):
    driver.get(link)
    loader = WebBaseLoader(link)
    return loader.load()


