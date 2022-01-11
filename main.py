from get_data import get_data
from machine_learning import test_models, supervised_machine_learning, unsupervised_machine_learning

if __name__ == '__main__':
    while True:
        print("""Choose an option:
        1. get data
        2. test models
        3. supervised machine learning
        4. unsupervised machine learning
        0. exit """)
        option = input("Enter an option (only number): ")
        if option == "1":
            get_data()
        elif option == "2":
            filename = input("filename: ")
            if not filename:
                filename = "data/learning_data.csv"
            test_models(filename)
        elif option == "3":
            filename1 = input("filename for learn: ")
            filename2 = input("filename with web scraping data: ")
            if not filename1:
                filename1 = "data/learning_data.csv"
            if not filename2:
                filename2 = "data/web_scraping_data.csv"
            supervised_machine_learning(filename1, filename2)
        elif option == "4":
            filename1 = input("filename for learn: ")
            filename2 = input("filename with web scraping data: ")
            if not filename1:
                filename1 = "data/learning_data.csv"
            if not filename2:
                filename2 = "data/web_scraping_data.csv"
            unsupervised_machine_learning(filename1, filename2)
        elif option == "0":
            break
        else:
            print("Wrong option selected")
        print()
