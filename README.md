# CSCI 8360 - Project 2 - Cilia Segmentation

This project is implemented as a part of the Data Science Practicum (CSCI 8360) course at the University of Georgia, Spring 2019.
The goal was to develop a video segmentation pipeline that identifies the regions of the videos containing cilia as accurately as possible.

Please refer [Wiki](https://github.com/dsp-uga/Team-thweatt-p2/wiki) for more details on our approach.

## Getting Started 

The following instructions will assist you get this project running on your local machine for developing and testing purpose.

### Prerequisites:

1. Python: <br />
    To install Python, go [here](https://www.python.org/downloads/)
    
2. Tensorflow: <br />
    If you don't have it installed, [download Tensorflow here](https://www.tensorflow.org/install).

3. Opencv: <br />
    `pip install opencv-python` 

4. Pandas: <br />
    `pip install panadas`

5. Scikit-learn: <br />
    `pip install scikit-learn`

6. [Google cloud platform](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Google-Cloud-Platform-set-up)


### Data Preparation:
The following data files/folders are all available on GCP bucket: `gs://uga-dsp/project2` <br />
train.txt, test.txt, data directory, masks directory

Download these files into your project directory using gsutil:<br />
`gsutil cp -r gs://uga-dsp/project2/* base_dir`

### Run Instruction:
You may install the package using `pip` as follows:

`$ pip install --user -i https://test.pypi.org/simple/ thweatt`

In this case you can import the package and call different methods as follows:

`>>> import thweatt`

`>>> thweatt.process_train(base_dir, dim_x=640, dim_y=640, n_frames=30, pre_type='none')`

`>>> thweatt.process_test(base_dir, dim_x=640, dim_y=640, n_frames=30, pre_type='none')`

`>>> X_train, y_train, X_test, test_names = thweatt.load_data(base_dir, n_frames=30)`

`>>> fitted_model = thweatt.model_train(X_train=X_train, y_train=y_train, clf='rf')`

`>>> thweatt.model_predict(base_dir, X_test=X_test, test_names=test_names, trained_model=fitted_model)`

As a result of running the above code, output masks will be saved to base_dir/output/ directory.

Alternatively, you can download the source code and simply run the following command:

`$ python3 main.py --base_dir /path/to/project/directory/`

List of command line arguments to pass to the program are as follows:

	--base_dir: absolute project directory path.
	--clf: type of classifier to use. Current choices are 'rf' and 'svm'.
	--xdim: width of the images after preprocessing.
	--ydim: length of the images after preprocessing.
	--n_frames: number of the frames per video to consider.
	--pre_type: type of preprocessing. Choices are 'none', 'resize', or 'zero'.

The only reqired argument is the `base_dir` which is the directory containing `train_file`, `test_file`, `data\`, and `masks\`.

The see the above list in command line execute the following command:

`$ python3 main.py -h`

One typical usage is:

`$ python3 main.py --base_dir="../dataset/" --clf="rf" --xdim=640 --ydim=640 --n_frames=30 --pre_type="none"`


### Approach:

- [U-net](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Model-Approaches)
- [Random Forest](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Model-Approaches)
- [SVM](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Model-Approaches)

### Future Work:
- [Tiramisu](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Future-Work)
- [DenseNet](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Future-Work)

## Ethics Considerations
This project can be used as a part of a bigger project to detect and identify the effects of Cilia movement/changes that can be used for future Medical Research. This project was trained on Medical images and should only be used for detecting cilia movement from the video pipeline. 


## Contributing

The `master` branch of this repo is write-protected and every pull request must passes a code review before being merged.
Other than that, there are no specific guidelines for contributing.
If you see something that can be improved, please send us a pull request!

## Authors
(Ordered alphabetically)

- **Abhishek Chatrath**
- **Hemanth Dandu**
- **Saed Rezayi**
- **Vishakha Atole** 


See the [CONTRIBUTORS.md](https://github.com/dsp-uga/Team-thweatt-p2/blob/master/CONTRIBUTORS.md) file for details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/dsp-uga/Team-thweatt-p2/blob/master/LICENSE) file for details
