import setuptools

with open("README.md", 'r') as fp:
	long_description = fp.read()

setuptools.setup(
	name = "thweatt",
	version = "1.0.4",
	author="Saed Rezayi, Hemanth Dandu, Vishakha Atole, Abhishek Chatrath",
	author_email="saedr@uga.edu, hemanthme22@gmail.com, va45686@uga.edu, ac06389@uga.edu",
	license='MIT',
	description="A package for cilia segmentation task.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/dsp-uga/Team-thweatt-p2",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
	],
	test_suite='nose.collector',
	tests_require=['nose'],
    install_requires=['keras'],
)
