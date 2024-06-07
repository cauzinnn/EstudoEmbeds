import openai
import os
import pandas as pd 
import tiktoken
import ast
import numpy as np

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

df = pd.read_csv("data/stoicism.csv")

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

#pra contar tokens
def num_tokens(text: str, model: str = GPT_MODEL):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def embedding_query(query:str) -> list:
	response_embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
	query_embedding = response_embedding.data[0].embedding
	df = pd.DataFrame({"text": query, "embedding": query_embedding})
	df.to_csv("data/prompt.csv", index=False)
	return query_embedding

def indice_artigos_rankeados(query_embedding: list, embeddings) -> list:
	distancias = []
	for embedding in embeddings:
		distancia = np.dot(query_embedding, embedding)/(np.linalg.norm(query_embedding)*np.linalg.norm(embedding))
		distancias.append(distancia)
	indice_ordenados_distancias = np.argsort(distancias)
	return (indice_ordenados_distancias, distancias)

def ask(query, df, n_embeddings: int = 5) -> str:
	"""Answers a query using GPT and a dataframe of relevant texts and embeddings."""
	df['embedding'] = df['embedding'].apply(ast.literal_eval)  #voltar a usar lista ao invés de str
	embeddings = df['embedding']
	query_embedding = embedding_query(query=query)
	indices_rankeados, distancias = indice_artigos_rankeados(query_embedding=query_embedding, embeddings=embeddings)
	k_counter = 0
	strings = []
	for i in indices_rankeados:
		if k_counter >= n_embeddings:
			break
		k_counter += 1
		string_final = str(df["text"][i][:80]).replace("\n", "")
		strings.append(str(df["text"][i]))
		print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {n_embeddings}) ---
        String: {string_final}
        Distance: {distancias[i]:0.3f}""")
	intro = f"Use the below articles on stoicism to answer the subsequent question. If the answer cannot be found in the articles, write \"I could not find an answer.\""
	pergunta = f"\n\nQuestion: {query}"
	artigos = "\n\nWikipedia article section\n\n" + "\n\n".join(strings)
	#message = query_message(query, df, model=model, token_budget=token_budget)
	query = intro + pergunta + artigos
	messages = [
        {"role": "system", "content": "You answer questions about stoicism"},
        {"role": "user", "content": query},
    ]
	response = client.chat.completions.create(model=GPT_MODEL, messages=messages, temperature=0)
	response_message = response.choices[0].message.content
	print(messages)
	print(response_message)
	return response_message

ask('Who were the main opponents of Stoicism?', df=df, n_embeddings=5)
