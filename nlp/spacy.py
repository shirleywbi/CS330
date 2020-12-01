import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_md")
nlp_obama = nlp('Obama speaks to the media in Illinois')
nlp_president = nlp('The president greets the press in Chicago')
nlp_unrelated = nlp('Data science is a multidisciplinary blend of data inference, algorithmm development, and technology.')

nlp_obama.similarity(nlp_president)
nlp_obama.similarity(nlp_unrelated)

# Getting entities
nlp_obama.ents
nlp_president.ents
nlp_unrelated.ents

displacy.render(nlp_obama, style="ent", jupyter=True)
displacy.render(nlp(df_train.iloc[10]["review"]), jupyter=True, style='ent')

for ent in nlp(df_train.iloc[10]["review"]).ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

displacy.render(nlp("Apple is a delicious fruit. I got my computer from Apple while I was eating an apple"), style="ent", jupyter=True)