import pyterrier as pt
import pandas as pd
import os
import warnings
import lightgbm as lgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


if __name__ == '__main__' :
    if not pt.started():
        pt.init()

    print('Load dataset')
    df = pd.read_csv('met_dataset.csv', dtype=str).astype(str).rename(columns={'Object ID': 'docno'})
    print('Dataset loaded successfully')

    index_path = './index'
    if not os.path.exists(index_path + "/data.properties"):
        print('Indexing artworks')
        pd_indexer = pt.DFIndexer(index_path, overwrite=True, blocks=True, verbose=True)
        meta_fields = df[['docno', 'Is Highlight', 'Is Timeline Work', 'Department', 'Title', 'Culture', 'Period', 'Artist Display Name', 'Country', 'Tags', 'image']]
        indexref = pd_indexer.index(df["description"], **meta_fields)
    else:
        print('Use existing index')
        indexref = pt.IndexRef.of(index_path + '/data.properties')
    index = pt.IndexFactory.of(indexref)

    topics = pd.read_csv('topics.csv').astype(str)
    qrels = pd.read_csv('qrels.csv').astype(str)
    qrels['label'] = qrels['label'].astype(int)

    RANK_CUTOFF = 50
    SEED=42

    # tr_va_topics, test_topics = train_test_split(topics, test_size=0, random_state=SEED)
    train_topics, valid_topics =  train_test_split(topics, test_size=0.35, random_state=SEED)

    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    sdm = pt.rewrite.SDM()
    qe = pt.rewrite.Bo1QueryExpansion(index)

    ltr_feats1 = (bm25 % RANK_CUTOFF) >> pt.text.get_text(index, ['Is Highlight', 'Is Timeline Work', 'Department', 'Title', 'Culture', 'Period', 'Artist Display Name', 'Country', 'Tags', 'image']) >> (
        pt.transformer.IdentityTransformer()
        ** # sequential dependence and query expansion
        (sdm >> bm25 >> qe >> bm25)
        ** # score of title (not originally indexed)
        (pt.text.scorer(body_attr="Title", takes='docs', wmodel="DirichletLM") ) 
        ** # score of author (not originally indexed)
        (pt.text.scorer(body_attr="Artist Display Name", takes='docs', wmodel="CoordinateMatch") ) 
        ** # score of tags (not originally indexed)
        (pt.text.scorer(body_attr="Tags", takes='docs', wmodel="DirichletLM") ) 
        ** # score of Country (not originally indexed)
        (pt.text.scorer(body_attr="Country", takes='docs', wmodel="DirichletLM") ) 
        ** # score of Department (not originally indexed)
        (pt.text.scorer(body_attr="Department", takes='docs', wmodel="DirichletLM") ) 
        ** # score of Culture (not originally indexed)
        (pt.text.scorer(body_attr="Culture", takes='docs', wmodel="DirichletLM") ) 
        ** # score of Period (not originally indexed)
        (pt.text.scorer(body_attr="Period", takes='docs', wmodel="DirichletLM") ) 
        ** # is highlited
        (pt.apply.doc_score(lambda row: int(row["Is Highlight"] == 'True')))
        ** # is Timeline Work
        (pt.apply.doc_score(lambda row: int(row["Is Timeline Work"] == 'True')))
        ** # has image
        (pt.apply.doc_score(lambda row: int( row["image"] == '1' and len(row["image"]) > 0) ))
        ** # Dichichlet Language Model
        pt.BatchRetrieve(index, wmodel="DirichletLM")
    )

    # for reference, lets record the feature names here too
    fnames=["BM25", "SDM and QE", "Title", "Artist Name", "Tags", "Country", "Department", "Culture", "Period", "Is Highlight", "Is Timeline Work", "hasImage" , "DirichletLM"]

    # this configures LightGBM as LambdaMART
    lmart_l = lgb.LGBMRanker(
        task="train",
        silent=False,
        min_data_in_leaf=1,
        min_sum_hessian_in_leaf=1,
        max_bin=255,
        num_leaves=31,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        ndcg_at=[10],
        eval_at=[10],
        learning_rate= .1,
        importance_type="gain",
        num_iterations=100,
        early_stopping_rounds=5,
        verbosity=-1
    )

    lmart_x_pipe = ltr_feats1 >> pt.ltr.apply_learned_model(lmart_l, form="ltr", fit_kwargs={'eval_at':[10]})
    print('Training...')
    lmart_x_pipe.fit(train_topics, qrels, valid_topics, qrels)

    while True:
        user_query = input('Search for artworks (Enter q to exit): ')
        if user_query == 'q':
            break
        results = lmart_x_pipe.search(user_query)
        print('The top 5 related artworks are:')
        for i in range(5):
            print('\t{}, Link: https://www.metmuseum.org/art/collection/search/{}\n'.format(results['Title'].iloc[i], results['docno'].iloc[i]))
