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
	intro = f"Use the below articles on stoicism to answer the subsequent question. If the answer cannot be found in the articles, write \"I could not find an answer.\". Do not answer the question if it cannot be found in the articles."
	pergunta = f"\n\nQuestion: {query}"
	artigos = "\n\nWikipedia article section\n\n" + "\n\n".join(strings)
	#message = query_message(query, df, model=model, token_budget=token_budget)
	query = intro + artigos + pergunta
	messages = [
        {"role": "system", "content": "You answer questions about stoicism"},
        {"role": "user", "content": query},
    ]
	response = client.chat.completions.create(model=GPT_MODEL, messages=messages, temperature=0)
	response_message = response.choices[0].message.content
	print(messages)
	print(response_message)
	return response_message

embedding_query("Stoicism is a school of Hellenistic philosophy that flourished in Ancient Greece and Ancient Rome. The Stoics believed that the practice of virtue is enough to achieve eudaimonia: a well-lived life. The Stoics identified the path to achieving it with a life spent practicing the four virtues in everyday life: wisdom, courage, temperance or moderation, and justice, and living in accordance with nature. It was founded in the ancient Agora of Athens by Zeno of Citium around 300 BC. \n\
	Alongside Aristotle's ethics, the Stoic tradition forms one of the major founding approaches to virtue ethics. The Stoics are especially known for teaching that \"virtue is the only good\" for human beings, and that external things, such as health, wealth, and pleasure, are not good or bad in themselves (adiaphora) but have value as \"material for virtue to act upon\". Many Stoics—such as Seneca and Epictetus—emphasized that because \"virtue is sufficient for happiness\", a sage would be emotionally resilient to misfortune. The Stoics also held that certain destructive emotions resulted from errors of judgment, and they believed people should aim to maintain a will (called prohairesis) that is \"in accordance with nature\". Because of this, the Stoics thought the best indication of an individual's philosophy was not what a person said but how a person behaved. To live a good life, one had to understand the rules of the natural order since they believed everything was rooted in nature.\n\
	Stoicism flourished throughout the Roman and Greek world until the 3rd century AD, and among its adherents was Roman Emperor Marcus Aurelius. It experienced a decline after Christianity became the state religion in the 4th century AD. Since then, it has seen revivals, notably in the Renaissance (Neostoicism) and in the contemporary era (modern Stoicism).\n\n\
	History: The name Stoicism derives from the Stoa Poikile (Ancient Greek: ἡ ποικίλη στοά), or \"painted porch\", a colonnade decorated with mythic and historical battle scenes on the north side of the Agora in Athens where Zeno of Citium and his followers gathered to discuss their ideas, near the end of the 4th century BC. Unlike the Epicureans, Zeno chose to teach his philosophy in a public space. Stoicism was originally known as Zenonism. However, this name was soon dropped, likely because the Stoics did not consider their founders to be perfectly wise and to avoid the risk of the philosophy becoming a cult of personality.\n\
	Zeno's ideas developed from those of the Cynics (brought to him by Crates of Thebes), whose founding father, Antisthenes, had been a disciple of Socrates. Zeno's most influential successor was Chrysippus, who followed Cleanthes as leader of the school, and was responsible for molding what is now called Stoicism. Stoicism became the foremost popular philosophy among the educated elite in the Hellenistic world and the Roman Empire to the point where, in the words of Gilbert Murray, \"nearly all the successors of Alexander professed themselves Stoics\". Later Roman Stoics focused on promoting a life in harmony within the universe within which we are active participants. \n\
	Scholars usually divide the history of Stoicism into three phases: the Early Stoa, from Zeno's founding to Antipater; the Middle Stoa, including Panaetius and Posidonius; and the Late Stoa, including Musonius Rufus, Seneca, Epictetus, and Marcus Aurelius. No complete works survived from the first two phases of Stoicism. Only Roman texts from the Late Stoa survived.\n\n\
	Philosophical system: Philosophy does not promise to secure anything external for man, otherwise it would be admitting something that lies beyond its proper subject-matter. For as the material of the carpenter is wood, and that of statuary bronze, so the subject-matter of the art of living is each person's own life.\n\
	Of all the schools of ancient philosophy, Stoicism made the greatest claim to being utterly systematic. In the view of the Stoics, philosophy is the practice of virtue, and virtue, the highest form of which is utility, is generally speaking, constructed from ideals of logic, monistic physics, and naturalistic ethics. These three ideals constitute virtue which is necessary for 'living a well reasoned life', seeing as they are all parts of a logos, or philosophical discourse, which includes the mind's rational dialogue with itself. Of them, the Stoics emphasized ethics as the main focus of human knowledge, though their logical theories were of more interest for later philosophers.\n\
	Stoicism teaches the development of self-control as a means of overcoming destructive emotions; the philosophy holds that becoming a clear and unbiased thinker allows one to understand the universal reason (logos). Stoicism's primary aspect involves improving the individual's ethical and moral well-being: \"Virtue consists in a will that is in agreement with Nature\". This principle also applies to the realm of interpersonal relationships; \"to be free from anger, envy, and jealousy\", and to accept even slaves as \"equals of other men, because all men alike are products of nature\".\n\n\
	The Stoic ethic espouses a deterministic perspective; in regard to those who lack Stoic virtue, Cleanthes once opined that the wicked man is \"like a dog tied to a cart, and compelled to go wherever it goes\". A Stoic of virtue, by contrast, would amend his will to suit the world and remain, in the words of Epictetus, \"sick and yet happy, in peril and yet happy, dying and yet happy, in exile and happy, in disgrace and happy\", thus positing a \"completely autonomous\" individual will and at the same time a universe that is \"a rigidly deterministic single whole\". This viewpoint was later described as \"Classical Pantheism\" (and was adopted by Dutch philosopher Baruch Spinoza).\n\n\
	Logic: Diodorus Cronus, who was one of Zeno's teachers, is considered the philosopher who first introduced and developed an approach to logic now known as propositional logic, which is based on statements or propositions, rather than terms, differing greatly from Aristotle's term logic. Later, Chrysippus developed a system that became known as Stoic logic and included a deductive system, Stoic Syllogistic, which was considered a rival to Aristotle's Syllogistic (see Syllogism). New interest in Stoic logic came in the 20th century, when important developments in logic were based on propositional logic. Susanne Bobzien wrote, \"The many close similarities between Chrysippus's philosophical logic and that of Gottlob Frege are especially striking\".\n\
	Bobzien also notes that, \"Chrysippus wrote over 300 books on logic, on virtually any topic logic today concerns itself with, including speech act theory, sentence analysis, singular and plural expressions, types of predicates, indexicals, existential propositions, sentential connectives, negations, disjunctions, conditionals, logical consequence, valid argument forms, theory of deduction, propositional logic, modal logic, tense logic, epistemic logic, logic of suppositions, logic of imperatives, ambiguity and logical paradoxes\".\n\
	The Stoics held that all beings (ὄντα)—though not all things (τινά)—are material. Besides the existing beings they admitted four incorporeals (asomata): time, place, void, and sayable. They were held to be just 'subsisting' while such a status was denied to universals. Thus, they accepted Anaxagoras's idea (as did Aristotle) that if an object is hot, it is because some part of a universal heat body had entered the object. But, unlike Aristotle, they extended the idea to cover all accidents. Thus, if an object is red, it would be because some part of a universal red body had entered the object.\n\n\
	They held that there were four categories:\nSubstance (ὑποκείμενον): The primary matter, formless substance, (ousia) that things are made of\nQuality (ποιόν): The way matter is organized to form an individual object; in Stoic physics, a physical ingredient (pneuma: air or breath), which informs the matter\nSomehow disposed (πως ἔχον): Particular characteristics, not present within the object, such as size, shape, action, and posture\nSomehow disposed in relation to something (πρός τί πως ἔχον): Characteristics related to other phenomena, such as the position of an object within time and space relative to other objects\n\
	The Stoics outlined that our own actions, thoughts, and reactions are within our control. The opening paragraph of the Enchiridion states the categories as: \"Some things in the world are up to us, while others are not. Up to us are our faculties of judgment, motivation, desire, and aversion. In short, whatever is our own doing.\" These suggest a space that is up to us or within our power. \n\
	Epistemology: The Stoics propounded that knowledge can be attained through the use of reason. Truth can be distinguished from fallacy—even if, in practice, only an approximation can be made. According to the Stoics, the senses constantly receive sensations: pulsations that pass from objects through the senses to the mind, where they leave an impression in the imagination (phantasiai) (an impression arising from the mind was called a phantasma).\nThe mind has the ability to judge (συγκατάθεσις, synkatathesis)—approve or reject—an impression, enabling it to distinguish a true representation of reality from one that is false. Some impressions can be assented to immediately, but others can achieve only varying degrees of hesitant approval, which can be labeled belief or opinion (doxa). It is only through reason that we gain clear comprehension and conviction (katalepsis). Certain and true knowledge (episteme), achievable by the Stoic sage, can be attained only by verifying the conviction with the expertise of one's peers and the collective judgment of humankind.\n\nPhysics: According to the Stoics, the Universe is a material reasoning substance (logos), which was divided into two classes: the active and the passive.[26] The passive substance is matter, which \"lies sluggish, a substance ready for any use, but sure to remain unemployed if no one sets it in motion\". The active substance is an intelligent aether or primordial fire, which acts on the passive matter.\n\
	Everything is subject to the laws of Fate, for the Universe acts according to its own nature, and the nature of the passive matter it governs. The souls of humans and animals are emanations from this primordial Fire, and are, likewise, subject to Fate. Individual souls are perishable by nature, and can be \"transmuted and diffused, assuming a fiery nature by being received into the seminal reason (\"logos spermatikos\") of the Universe\". Since right Reason is the foundation of both humanity and the universe.\n\nStoic theology is a fatalistic and naturalistic pantheism: God is never fully transcendent but always immanent, and identified with Nature. Abrahamic religions personalize God as a world-creating entity, but Stoicism equates God with the totality of the universe; according to Stoic cosmology, which is very similar to the Hindu conception of existence, there is no absolute start to time, as it is considered infinite and cyclic. Similarly, space and the Universe have neither start nor end, rather they are cyclical. The current Universe is a phase in the present cycle, preceded by an infinite number of Universes, doomed to be destroyed (\"ekpyrōsis\", conflagration) and re-created again, and to be followed by another infinite number of Universes. Stoicism considers all existence as cyclical, the cosmos as eternally self-creating and self-destroying (see also Eternal return).\nStoicism does not posit a beginning or end to the Universe. According to the Stoics, the logos was the active reason or anima mundi pervading and animating the entire Universe. It was conceived as material and is usually identified with God or Nature. The Stoics also referred to the seminal reason (\"logos spermatikos\"), or the law of generation in the Universe, which was the principle of the active reason working in inanimate matter. Humans, too, each possess a portion of the divine logos, which is the primordial Fire and reason that controls and sustains the Universe.\n\nEthics: The foundation of Stoic ethics is that good lies in the state of the soul itself, in wisdom and self-control. One must therefore strive to be free of the passions. For the Stoics, reason meant using logic and understanding the processes of nature—the logos or universal reason, inherent in all things. The Greek word pathos was a wide-ranging term indicating an infliction one suffers. The Stoics used the word to discuss many common emotions such as anger, fear and excessive joy. A passion is a disturbing and misleading force in the mind which occurs because of a failure to reason correctly. \nFor the Stoic Chrysippus, the passions are evaluative judgements. A person experiencing such an emotion has incorrectly valued an indifferent thing. A fault of judgement, some false notion of good or evil, lies at the root of each passion. Incorrect judgement as to a present good gives rise to delight, while lust is a wrong estimate about the future. Unreal imaginings of evil cause distress about the present, or fear for the future. The ideal Stoic would instead measure things at their real value, and see that the passions are not natural. To be free of the passions is to have a happiness which is self-contained. There would be nothing to fear—for unreason is the only evil; no cause for anger—for others cannot harm you.\n\n\
	Passions: The Stoics arranged the passions under four headings: distress, pleasure, fear and lust. One report of the Stoic definitions of these passions appears in the treatise On Passions by Chrysippus (trans. Long & Sedley, pg. 411, modified):\nDistress (lupē): Distress is an irrational contraction, or a fresh opinion that something bad is present, at which people think it right to be depressed.\nFear (phobos): Fear is an irrational aversion, or avoidance of an expected danger.\nLust (epithumia): Lust is an irrational desire, or pursuit of an expected good but in reality bad.\nDelight (hēdonē): Delight is an irrational swelling, or a fresh opinion that something good is present, at which people think it right to be elated.\nTwo of these passions (distress and delight) refer to emotions currently present, and two of these (fear and lust) refer to emotions directed at the future.[39] Thus there are just two states directed at the prospect of good and evil, but subdivided as to whether they are present or future:[40] Numerous subdivisions of the same class were brought under the head of the separate passions:[41]\nDistress: Envy, Rivalry, Jealousy, Compassion, Anxiety, Mourning, Sadness, Troubling, Grief, Lamenting, Depression, Vexation, Despondency.\nFear: Sluggishness, Shame, Fright, Timidity, Consternation, Pusillanimity, Bewilderment, and Faintheartedness.\nLust: Anger, Rage, Hatred, Enmity, Wrath, Greed, and Longing.\nDelight: Malice, Rapture, and Ostentation.\n\
	The wise person (sophos) is someone who is free from the passions (apatheia). Instead the sage experiences good-feelings (eupatheia) which are clear-headed.[42] These emotional impulses are not excessive, but nor are they diminished emotions.[43][44] Instead they are the correct rational emotions.[44] The Stoics listed the good-feelings under the headings of joy (chara), wish (boulesis), and caution (eulabeia).[36] Thus if something is present which is a genuine good, then the wise person experiences an uplift in the soul—joy (chara).[45] The Stoics also subdivided the good-feelings:[46]

Joy: Enjoyment, Cheerfulness, Good spirits
Wish: Good intent, Goodwill, Welcoming, Cherishing, Love
Caution: Moral shame, Reverence
Suicide
The Stoics accepted that suicide was permissible for the wise person in circumstances that might prevent them from living a virtuous life,[47] such as if they fell victim to severe pain or disease,[47] but otherwise suicide would usually be seen as a rejection of one's social duty.[48] For example, Plutarch reports that accepting life under tyranny would have compromised Cato's self-consistency (constantia) as a Stoic and impaired his freedom to make the honorable moral choices.[49]


")
ask('Who were the main opponents of Apatheia and Stoicism?', df=df, n_embeddings=3)