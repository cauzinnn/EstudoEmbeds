import pandas as pd
import numpy as np
from ast import literal_eval
import scipy


df = pd.read_csv("data/stoicism.csv")


def print_recommendations_from_strings(
    df, index_procura: int, k_nearest_neighbors: int = 5
) -> list[int]:
    relatedness_fn = lambda x, y: scipy.spatial.distance.cosine(x, y)

    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
    embeddings = df["embedding"]
    distances = []
    for embedding in embeddings:
        distancia = relatedness_fn(embeddings[index_procura], embedding)
        distances.append(distancia)

    indice_das_distancias = np.argsort(distances)

    query_string = embeddings[index_procura]

    k_counter = 0
    for i in indice_das_distancias:
        if np.all(query_string == df["embedding"][i]):
            continue
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        string_final = str(df["text"][i][:80]).replace("\n", " ")
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {string_final}
        Distance: {distances[i]:0.3f}"""
        )

    return indice_das_distancias


teste = print_recommendations_from_strings(
    df=df,
    index_procura=5,
    k_nearest_neighbors=10,
)
