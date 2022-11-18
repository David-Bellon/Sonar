import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    df = pd.read_csv("sonar-data.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(df.drop(columns=["R"]), df["R"], train_size=0.7, random_state=69)
    return X_train, X_test, Y_train, Y_test