import pandas as pd
import bnlearn as bn
import sys
from bayesian_generator import BayesianGenerator
from input_data import probabilities, edges, example_name


def simulation(data_samples: int = 1000000):
    """
    Function creates a model with probabilites from input_data.py.
    Probabilities should be organised:
    T
    FF, TF, FT, TT
    FFF, TFF, FTF, TTF, FFT, TFT, FTT, TTT
    etc ...

    param data_samples: number of data samples that must be generated
    type data_samples: int
    """
    generator = BayesianGenerator(probabilities)
    variables, data = generator.generate_data(data_samples)
    df = pd.DataFrame(columns=variables, data=data)

    DAG = bn.make_DAG(edges)
    model = bn.parameter_learning.fit(DAG, df, verbose=0)
    bn.print_CPD(model)


def run():
    """
    Funcion runs simulation for several hardcoded number of samples.
    """
    for data_samples in (1000, 10000, 50000, 100000, 500000, 1000000):
        # inform user
        sys.stdout = sys.__stdout__
        print(f'Simalation with {data_samples} data samples has started.')
        # run simulation and save results to file
        sys.stdout = open(
            f'examplary_results/{example_name}_{data_samples}_samples.txt', 'w'
        )
        simulation(data_samples)
        sys.stdout.close()
        # inform user
        sys.stdout = sys.__stdout__
        print(f'Simalation results of {data_samples} data samples saved.')


if __name__ == '__main__':
    run()
