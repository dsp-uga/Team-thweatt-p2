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


### Approach:

- [U-net](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Model-Approaches)
- [Random Forest](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Model-Approaches)
- [SVM](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Model-Approaches)

### Future Work:
- [Tiramisu](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Future-Work)
- [DenseNet](https://github.com/dsp-uga/Team-thweatt-p2/wiki/Future-Work)

## Ethics
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
