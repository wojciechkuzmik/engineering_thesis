import pandas as pd
import requests
from bs4 import BeautifulSoup


def scrap_pages(olx_link, ceneo_link):
    page = requests.get(ceneo_link)
    soup = BeautifulSoup(page.content.decode("utf-8"), "html.parser")
    ceneo_data = [x['data-productname'] for x in soup.findAll("div", class_="cat-prod-row js_category-list-item "
                                                                            "js_clickHashData js_man-track-event")]
    page = requests.get(olx_link)
    soup = BeautifulSoup(page.content.decode("utf-8"), "html.parser")
    olx_data = [result.strong.string for result in soup.find_all(class_="title-cell")]
    length = min(len(olx_data), len(ceneo_data))
    d = {"first": ceneo_data[:length], "second": olx_data[:length], "match": 1}
    df = pd.DataFrame(d)
    return df


def get_data():
    links = [
        ("https://www.olx.pl/elektronika/telefony/smartfony-telefony-komorkowe/samsung/q-s21/",
         "https://www.ceneo.pl/Smartfony;szukaj-samsung+s21"),
        ("https://www.olx.pl/elektronika/telefony/smartfony-telefony-komorkowe/iphone/q-12-pro-max/",
         "https://www.ceneo.pl/Smartfony;szukaj-iphone+12+pro+max"),
        ("https://www.olx.pl/elektronika/telefony/smartfony-telefony-komorkowe/iphone/q-13-mini/",
         "https://www.ceneo.pl/Smartfony;szukaj-iphone+13+mini")
    ]
    df = pd.DataFrame(columns=["first", "second", "match"])
    for link in links:
        df = df.append(scrap_pages(link[0], link[1]))

    df.to_csv("data/web_scraping_data.csv", index=False)
