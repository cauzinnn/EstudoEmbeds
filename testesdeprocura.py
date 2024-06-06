import pandas as pd
import numpy as np
from ast import literal_eval


df = pd.read_csv("data/stoicism.csv")

def print_recommendations_from_strings(
    df,
    index_procura: int,
    k_nearest_neighbors: int = 5) -> list[int]:
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
    embeddings = df["embedding"]
    distances = []
    for embedding in embeddings:
    	a = np.dot(embeddings[index_procura], embedding)/(np.linalg.norm(embeddings[index_procura])*np.linalg.norm(embedding))
    	distances.append(a)

    indice_das_distancias = np.argsort(distances)

    query_string = embeddings[index_procura]

    k_counter = 0
    for i in indice_das_distancias:
        if np.all(query_string == df["embedding"][i]):
            continue
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1



        string_final = str(df["text"][i][:80]).replace('\n',' ')
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {string_final}
        Distance: {distances[i]:0.3f}"""
        )

    return indice_das_distancias

teste = print_recommendations_from_strings(
    df=df, 
    index_procura=0, 
    k_nearest_neighbors=10, 
)