import argparse
import sys

from preprocessing import process_train, process_test
from model_builder import load_data, model_train, model_predict

def main():
    """ Main function
    """

    # Reading command line arguments
    parser = argparse.ArgumentParser(description='Read in file paths and other parameters.')
    parser.add_argument('--base_dir', help="absolute project directory path.", required=True, type=str)
    parser.add_argument('--clf', choices=['rf', 'svm'], help="type of classifier to use.", default='rf', type=str)
    parser.add_argument('--xdim', help="width of the images after preprocessing.", default=640, type=int)
    parser.add_argument('--ydim', help="length of the images after preprocessing.", default=640, type=int)
    parser.add_argument('--n_frames', help="number of the frames per video to consider.", default=30, type=int)
    parser.add_argument('--pre_type', choices=['none', 'resize', 'zero'], help="type of preprocessing.", default='none', type=str)
    args = parser.parse_args()

    print("Initializing the variables....")
    base_dir = args.base_dir
    clf = args.clf
    dim_x = args.xdim
    dim_y = args.ydim
    n_frames = args.n_frames
    pre_type = args.pre_type

    # Extracting, preprocessing, and saving training images and masks. 
    process_train(base_dir, dim_x=dim_x, dim_y=dim_y, n_frames=n_frames, pre_type=pre_type)

    # Extracting, preprocessing, and saving training images and masks. 
    process_test(base_dir, dim_x=dim_x, dim_y=dim_y, n_frames=n_frames, pre_type=pre_type)

    # Loading data into variables
    X_train, y_train, X_test = load_data(base_dir, n_frames=n_frames)

    # Training the classifier
    fitted_model = model_train(X_train=X_train, y_train=y_train, clf=clf)

    # Predicting the results
    model_predict(base_dir, X_test=X_test, trained_model=fitted_model)

if __name__ == '__main__':
    sys.exit(main())
