from get_data import get_data
from machine_learning import test_models, supervised_machine_learning, unsupervised_machine_learning, tune_gbclassifier, \
    show_data

default_learning_data_f = "data/data.csv"
default_web_scraping_data_f = "data/web_scraping_data.csv"

if __name__ == '__main__':
    while True:
        print("""Choose an option:
        1. get data
        2. test models
        3. supervised machine learning
        4. unsupervised machine learning
        5. tune GradientBoostClassifier
        6. show data
        0. exit """)
        option = input("Enter an option (only number): ")
        if option == "1":
            filename = input("filename to store data (press enter to choose default filename): ")
            if not filename:
                filename = default_web_scraping_data_f
            get_data(filename)
        elif option == "2":
            filename = input("filename with learning data (press enter to choose default filename): ")
            if not filename:
                filename = default_learning_data_f
            test_models(filename)
        elif option == "3":
            filename1 = input("filename with learning data (press enter to choose default filename): ")
            filename2 = input("filename with web scraping data (press enter to choose default filename): ")
            if not filename1:
                filename1 = default_learning_data_f
            if not filename2:
                filename2 = default_web_scraping_data_f
            supervised_machine_learning(filename1, filename2)
        elif option == "4":
            filename1 = input("filename with learning data (press enter to choose default filename): ")
            filename2 = input("filename with web scraping data (press enter to choose1 default filename): ")
            if not filename1:
                filename1 = default_learning_data_f
            if not filename2:
                filename2 = default_web_scraping_data_f
            unsupervised_machine_learning(filename1, filename2)
        elif option == "5":
            filename = input("filename with learning data (press enter to choose default filename): ")
            if not filename:
                filename = default_learning_data_f
            tune_gbclassifier(filename)
        elif option == "6":
            filename = input("filename with learning data (press enter to choose default filename): ")
            if not filename:
                filename = default_learning_data_f
            show_data(filename)
        elif option == "0":
            break
        else:
            print("Wrong option selected")
        print()
