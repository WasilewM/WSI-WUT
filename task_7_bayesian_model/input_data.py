# 1st example
example_name = 'example1'
probabilities = {
    'voltage_change': ([], [0.03]),
    'flooding': ([], [0.3]),
    'drive_failure': (
        ['voltage_change', 'flooding'],
        # voltage_change, flooding: FF, TF, FT, TT
        [0.01, 0.20, 0.70, 0.95]
    ),
    'computer_failure': ([], [0.09]),
    'data_loss': (
        ['drive_failure', 'computer_failure'],
        # drive_failure, computer_failure: FF, TF, FT, TT
        [0.001, 0.85, 0.15, 0.90]
    )
}

edges = [
    ('voltage_change', 'drive_failure'),
    ('flooding', 'drive_failure'),
    ('drive_failure', 'data_loss'),
    ('computer_failure', 'data_loss')
]

# # 2nd example
# example_name = 'example2'
# probabilities = {
#     'a': ([], [0.6]),
#     'b': ([], [0.4]),
#     'c': (
#         ['a', 'b'],
#         # a, b: FF, TF, FT, TT
#         [0.2, 0.65, 0.6, 0.85]
#     )
# }

# edges = [
#     ('a', 'c'),
#     ('b', 'c')
# ]
