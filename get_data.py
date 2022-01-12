import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np


def scrap_pages(olx_link, ceneo_link, brand):
    page = requests.get(ceneo_link)
    soup = BeautifulSoup(page.content.decode("utf-8"), "html.parser")
    ceneo_titles = [x['data-productname'] for x in soup.findAll("div", class_="cat-prod-row js_category-list-item "
                                                                              "js_clickHashData js_man-track-event")]
    ceneo_prices = [x['data-productminprice'] for x in soup.findAll("div", class_="cat-prod-row js_category-list-item "
                                                                                  "js_clickHashData js_man-track-event")]

    page = requests.get(olx_link)
    soup = BeautifulSoup(page.content.decode("utf-8"), "html.parser")
    olx_titles = [result.strong.string for result in soup.find_all(class_="title-cell")]
    olx_prices = [x.strong.string for x in soup.findAll("p", class_="price")]
    olx_prices_fixed = []
    for price in olx_prices:
        olx_prices_fixed.append(''.join([n for n in price if n.isdigit()]))
    length = min(len(olx_titles), len(ceneo_titles))
    d = {"first_title": ceneo_titles[:length], "first_price": ceneo_prices[:length],
         "second_title": olx_titles[:length],
         "second_price": olx_prices_fixed[:length], "label": 1}
    df = pd.DataFrame(d)
    df['first_brand'] = brand
    df['second_brand'] = brand
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[['first_title', 'first_brand', 'first_price', 'second_title', 'second_brand', 'second_price', 'label']]
    return df


def get_data(output_filename):
    links = [
        ("https://www.olx.pl/elektronika/telefony/smartfony-telefony-komorkowe/samsung/q-s21/",
         "https://www.ceneo.pl/Smartfony;szukaj-samsung+s21", "samsung"),
        ("https://www.olx.pl/elektronika/telefony/smartfony-telefony-komorkowe/iphone/q-12-pro-max/",
         "https://www.ceneo.pl/Smartfony;szukaj-iphone+12+pro+max", "iphone"),
        ("https://www.olx.pl/elektronika/telefony/smartfony-telefony-komorkowe/iphone/q-13-mini/",
         "https://www.ceneo.pl/Smartfony;szukaj-iphone+13+mini", "iphone")
    ]
    df = pd.DataFrame(columns=['first_title', 'first_brand', 'first_price', 'second_title', 'second_brand',
                               'second_price', 'label'])
    for link in links:
        df = df.append(scrap_pages(link[0], link[1], link[2]))

    df.to_csv(output_filename, index=False)
