import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, KeyedVectors, FastText

# Word2Vec
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
word_pairs = [('pineapple','mango'), ('pineapple','juice'), ('sun','robot')]
for pair in word_pairs: 
    print('The similarity between %s and %s is %0.3f' %(pair[0], pair[1], model.similarity(pair[0], pair[1])))

## Finding similar words
model.most_similar('UBC')

## Finding the odd one out
model.doesnt_match("sun moon earth UBC mars".split())

## Distance between sentences

sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
sentence_president = 'The president greets the press in Chicago'.lower().split()
sentence_unrelated = 'Data science is a multidisciplinary blend of data inference, algorithmm development, and technology.'

similarity = model.wmdistance(sentence_obama, sentence_president)
print("Distance between related sentences {:.4f}".format(similarity))

similarity = model.wmdistance(sentence_obama, sentence_unrelated)
print("Distance between unrelated sentences {:.4f}".format(similarity))

## Analogy
def analogy(word1, word2, word3):
    print('%s : %s :: %s : ?' %(word1, word2, word3))
    sim_words = model.most_similar(positive=[word3, word2], negative=[word1])
    return pd.DataFrame(sim_words, columns=['Analogy word', 'Score'])

analogy('man','king','woman')