import argparse


"""
Testing and evaluating the model.
"""


def test_model(test_data):
    # TODO:
    # 1. upload the trained model
    # 2. run the model with a test data
    # 3. using the inverted index estimate relevance
    # 4. report results

    print(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, help="Path to testing data")

    args = parser.parse_args()
    test_model(args.test_data)

