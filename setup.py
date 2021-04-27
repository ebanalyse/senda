import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SENDA", 
    version="0.7.1",
    author="Lars Kjeldgaard",
    author_email="lars.kjeldgaard@eb.dk",
    description="Framework for Fine-tuning Transformers for Sentiment Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebanalyse/SENDA",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    install_requires=[
        'torch',
        'transformers',
        'sklearn',
        'nltk',
        'pandas',
        'pyconll',
        'tweepy',
        'danlp',
        'datasets',
        'numpy'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest',
                   'pytest-cov'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
    )
