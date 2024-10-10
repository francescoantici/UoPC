import argparse
import pandas as pd

from feature_encoders.SBert import SBert
from predictive_models.KNN import KNN
from utils.utils import file_parser, read_user_dataset

if __name__ == "__main__":
    
    # Add parser for the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--user-dataset", type=str, required=True)
    parser.add_argument("-f", "--file", type=str, required=False)
    parser.add_argument("-v", "--values", type=str, required=False)
    
    args = parser.parse_args()
    
    # If the user dataset is not provided the prediction cannot be performed
    if not(args.user_dataset):
        print("Please provide a user dataset file")
        break
    
    # Init the components of the framework
    predictive_model = KNN()
    encoder = SBert()
    
    # Read the user data and build the predictive model
    udf = read_user_dataset(args.user_dataset) 
    predictive_model.train(udf.encodings, udf.pcon)
    
    # Retrieve the new job data
    if args.file:
        features = file_parser(args.file)
    elif args.values:
        features = args.values
    
    # Return the prediction
    new_job_encoding = encoder.encode_job(features)
    print(predictive_model.predict(new_job_encoding))
    
    