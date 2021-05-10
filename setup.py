import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="senda", 
    version="0.7.7",
    author="Lars Kjeldgaard",
    author_email="lars.kjeldgaard@eb.dk",
    description="Framework for Fine-tuning Transformers for Sentiment Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebanalyse/senda",
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
    )
