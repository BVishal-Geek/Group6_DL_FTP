import os

import pandas as pd

import argparse

def conversionTxtToCSV(input_path, output_path):

    """
    Converts a text file to a CSV file.
    Args:

    input_path (str): Path to the input text file.

    output_path (str): Path to the output CSV file.

    """

    try:

        if os.path.isdir(input_path):
            txtFile = os.path.join(input_path, 'List_of_testing_videos.txt')
        else:
            txtFile= input_path

        with open(txtFile, 'r') as file:
            data = file.readlines()

        df = pd.DataFrame([row.strip().split(' ') for row in data], columns=['target', 'videoname'])

        df.to_csv(output_path + '/' +'test_videos.csv', index=False)


    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert text file to CSV')

    parser.add_argument('input_path', type=str, help='Path to the input text file')

    parser.add_argument('output_path', type=str, help='Path to the output CSV file')

    args = parser.parse_args()

    conversionTxtToCSV(args.input_path, args.output_path)