import run_k_folded_cross_validation
import run_fit_test_split


def collect_fit_test_accuracy_data(n: int = 100):
    """
    Function performs collects accuracy tests results of a fit test approach.

    param n: number of tests
    type n: int
    """
    is_test_set_equal_fit_set = False
    accuracy_results = "data_proportion;accuracy;is_test_set_equal_fit_set;\n"
    for _ in range(2):
        for data_proportion in (0.1, 0.3, 0.6, 0.9):
            for _ in range(n):
                accuracy_results += str(data_proportion) + ";"
                accuracy_results += str(run_fit_test_split.run(
                    data_proportion, is_test_set_equal_fit_set))
                accuracy_results += f";{is_test_set_equal_fit_set};\n"
        is_test_set_equal_fit_set = True
    return accuracy_results


def collect_k_folded_validation_accuracy_data(n: int = 100):
    """
    Function performs collects accuracy tests results of a k-folded cross
    validation approach.

    param n: number of tests
    type n: int
    """
    accuracy_results = "k;accuracy;\n"
    for subsets_number in (2, 3, 5, 10):
        for _ in range(n):
            accuracy_results += str(subsets_number) + ";"
            accuracy_results += str(run_k_folded_cross_validation.run(
                subsets_number
            ))
            accuracy_results += ";\n"
    return accuracy_results


def save_data(file_path: str, data: str):
    """
    Function saves data into a given file.

    param file_path: path to the file
    type: str

    param data: collected data in a csv format
    type data: str
    """
    try:
        with open(file_path, "w") as file_handle:
            for line in data:
                file_handle.write(line)
            print("Saving data into given file completed.")
    except IsADirectoryError:
        print("Path is a directory.")
    except FileNotFoundError:
        print("File not found.")
    except Exception:
        print("Undetermined error occurred. Please try again.")


if __name__ == "__main__":
    accuracy_results = collect_fit_test_accuracy_data()
    save_data("fit_test_accuracy_results.csv", accuracy_results)
    accuracy_results = collect_k_folded_validation_accuracy_data()
    save_data("k_folded_validation_accuracy_results.csv", accuracy_results)
