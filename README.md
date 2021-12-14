# SI650-Project: Artwork Search Engine
This is the final project report for Umich SI650 Information Retrieval, where we implemented and improved the search engine for artworks in The Metropolitan Museum of Art.
## Requirements:
The codes rely on python v3.7.
You can install the dependencies by cd to the root directory and run `conda install -r requirements`.
## Learning to Rank:
The implmentation and evaluation can be found in the "learning2rank.ipynb". To run the search engine, simply run `python learning2rank.py`.
## Dense Retrieval:
This search engine utilize the power of language model to encode the documents and queries into dense vectors in the same vector space. You can simply run the search engine by `python dense_retrieval.py` in command line. It will prompt you to enter a query. It can return search results for 3 models: BM25, Bi-encoder retrieval and cross-encoder retrieval. The cross-encoder retrieval gives the best search results usually. It may cost some time for the first time since it needs to download language models.
