import csv


def read_data_from_csv(file_handle):
    """
    Funcion reads the data from provided file.
    """
    data = []
    reader = csv.DictReader(file_handle)
    try:
        for data_record in reader:
            new_data = [
                float(data_record["fixed acidity"]),
                float(data_record["volatile acidity"]),
                float(data_record["citric acid"]),
                float(data_record["residual sugar"]),
                float(data_record["chlorides"]),
                float(data_record["free sulfur dioxide"]),
                float(data_record["total sulfur dioxide"]),
                float(data_record["density"]),
                float(data_record["pH"]),
                float(data_record["sulphates"]),
                float(data_record["alcohol"]),
                int(data_record["quality"])
            ]
            data.append(new_data)
    except KeyError:
        print("Key Error occured - cannot load data set")
        return None
    except Exception:
        print("Unidentified error occurred - cannot load data set")
        return None

    return data


def get_data():
    """
    Function managesthe process of retrieving data from a file.
    """
    try:
        data_set = []
        with open("winequality-red.csv", "r") as file_handle:
            data_set = read_data_from_csv(file_handle)
        return data_set
    except IsADirectoryError:
        print("Path is a directory.")
    except FileNotFoundError:
        print("File not found.")
    except Exception:
        print("Undetermined error occurred. Please try again.")
