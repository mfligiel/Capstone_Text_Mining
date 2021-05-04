import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.2'
PACKAGE_NAME = 'capstone_text_mining'
AUTHOR = 'Matt Fligiel, Carmin Ballou, Zain Jafri'
AUTHOR_EMAIL = 'matthewfligiel@gmail.com'
URL = 'https://github.com/mfligiel/Capstone_Text_Mining'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Capstone Text Mining Techniques'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

#this installation doesn'thave the packages needed for carmin and zain's pieces yet.
INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'matplotlib',
      'seaborn',
      'tqdm',
      'sklearn',
      'nltk',
      'vaderSentiment',
      'spacy',
      'networkx',
      'gensim',
      'pyLDAvis==2.1.2', #needed for gensim
      'snorkel'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
