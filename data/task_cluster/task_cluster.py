import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np

def assign_single_cluster_semi_supervised_n(df, texts, reference_ids, ref_texts, ref_ids, n):
    vec = TfidfVectorizer(stop_words='english')

    combined_texts = list(texts) + [text for text in ref_texts if text not in texts]
    X = vec.fit_transform(combined_texts)

    main_count = len(texts)
    reference_idxs = list(range(main_count, main_count + len(ref_texts)))

    similarities = cosine_similarity(X[:main_count], X[reference_idxs])

    clusters = []
    referred = []
    referred_questions = []

    for row in similarities:
        top_indices = row.argsort()[-n:][::-1]  # Get indices of top n similarities in descending order
        clusters.append(str(top_indices.tolist()))
        referred.append(str([ref_ids[i] for i in top_indices]))
        referred_questions.append(' || '.join([ref_texts[i] for i in top_indices]))

    df = df.copy()  # ensure original is preserved
    df['clusters'] = clusters
    df['referred'] = referred
    df['referred_question'] = referred_questions

    return df

def assign_single_cluster_unsupervised(df_default, df_dev, n_clusters=10):
    combined_df = pd.concat([df_default.copy(), df_dev.copy()], ignore_index=True)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(combined_df['question'].astype(str).str.strip())

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    combined_df['cluster'] = kmeans.fit_predict(X)

    cluster_column = combined_df['cluster']

    # split cluster labels back
    df_default_out = df_default.copy()
    df_dev_out = df_dev.copy()

    df_default_out['cluster'] = cluster_column.iloc[:len(df_default)].values
    df_dev_out['cluster'] = cluster_column.iloc[len(df_default):].values

    # map cluster -> list of task_ids and questions in dev
    cluster_to_dev_tasks = df_dev_out.groupby('cluster')['task_id'].apply(list).to_dict()
    cluster_to_dev_questions = df_dev_out.groupby('cluster')['question'].apply(lambda x: ' || '.join(x)).to_dict()

    # assign both list of task_ids and concatenated question strings
    df_default_out['reference_list'] = df_default_out['cluster'].map(cluster_to_dev_tasks).apply(lambda x: x if isinstance(x, list) else [])
    df_default_out['referred_question'] = df_default_out['cluster'].map(cluster_to_dev_questions).fillna('')

    return df_default_out


def assign_multi_clusters(df, texts, reference_ids, ref_texts, ref_ids, threshold=0.2):
    vec = TfidfVectorizer(stop_words='english')

    combined_texts = list(texts) + [text for text in ref_texts if text not in texts]
    X = vec.fit_transform(combined_texts)

    main_count = len(texts)
    reference_idxs = list(range(main_count, main_count + len(ref_texts)))
    similarities = cosine_similarity(X[:main_count], X[reference_idxs])

    clusters = []
    referred = []
    referred_questions = []

    for row in similarities:
        indices = [i for i, sim in enumerate(row) if sim >= threshold]
        cluster_ids = indices
        task_ids = [ref_ids[i] for i in indices]
        questions = [ref_texts[i] for i in indices]

        clusters.append(str(cluster_ids))
        referred.append(str(task_ids))
        referred_questions.append(' || '.join(questions))

    df = df.copy()
    df['clusters'] = clusters
    df['referred'] = referred
    df['referred_question'] = referred_questions

    return df

def main():
    df = pd.read_csv('../../test_case/dabstep_default.csv')
    dev_df = pd.read_csv('../../test_case/dabstep_dev.csv')

    reference_ids = [5, 49, 1273, 1305, 1464, 1681, 1753, 1871, 2697]

    dev_ref_df = dev_df[dev_df['task_id'].isin(reference_ids)]
    ref_ids = dev_ref_df['task_id'].tolist()
    ref_texts = dev_ref_df['question'].astype(str).str.strip().tolist()
    texts = df['question'].astype(str).str.strip()
    
    n = 2
    df_single = assign_single_cluster_semi_supervised_n(df, texts, reference_ids, ref_texts, ref_ids, n=n)
    df_single.to_csv(f'clustered_single_semi_{n}.csv', index=False)

    df_unsupervised = assign_single_cluster_unsupervised(df, dev_df, n_clusters=10)
    df_unsupervised.to_csv('clustered_single_unsupervised.csv', index=False)

    df_multi = assign_multi_clusters(df, texts, reference_ids, ref_texts, ref_ids)
    df_multi.to_csv('clustered_multi_unsupervised.csv', index=False)

if __name__ == '__main__':
    main()
