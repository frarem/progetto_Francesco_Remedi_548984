import sys
import string
import nltk

#liste delle POS per le parole funzionli e quelle semanticamente piene
lista_POS_piene = ["NN", "NNS", "NNP", "JJ", "JJR", "JJS", "RB", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
lista_POS_funzionali = ["CC", "DT", "CD", "IN", "TO", "RP", "MD", "PRP", "PRP$", "WDT", "WP"]

def estraiTokens (frasi):
    #variabili per lunghezza del corpus e per contenere il corpus
    lenCorpus = 0
    corpus = []
    #ciclo che scorre frase per frase
    for frase in frasi:
        #da frase estraggo i tokens e li aggiungo alla variabile corpus
        tokens = nltk.word_tokenize(frase)
        corpus += tokens
    #numero dei token del file
    lenCorpus = len(corpus)
    #creo lista dei token DIVERSI e ordinati
    vocabolario = list(sorted(set(corpus)))
    return vocabolario, corpus, lenCorpus
    
def calcoloMedia(frasi):
    #liste in cui salvo la lunghezza in token delle frasi e le lunghezze in caratteri dei token
    lenFrasi = []
    lenTokens = []
    #ciclo che scorre frase per frase
    for frase in frasi:
        #elimino dalla frase la punteggiatura   https://www.w3schools.com/python/ref_string_maketrans.asp
        frase = frase.translate(str.maketrans("", "", string.punctuation))
        #da frase estraggo i tokens 
        tokens = nltk.word_tokenize(frase)
        #calcolo quanti token ci sono nella frase e salvo il numero in lenFrasi
        lenFrasi.append(len(tokens))
        #ciclo che scorre token per token
        for token in tokens:
            #calcolo lunghezza in caratteri di ogni token e salvo i valori in lenTokens
            lenTokens.append(len(token))
    #finiti i cicli calcolo la media sommando i valori contenuti nelle liste e dividendo per la loro lunghezza
    mediaTokenFrasi = sum(lenFrasi)/len(lenFrasi)
    mediaCaratteriTokens = sum(lenTokens)/len(lenTokens)
    return mediaTokenFrasi, mediaCaratteriTokens

def CalcoloHapax (vocabolario, corpus):
    hapax = []
    #scorro vocabolario un token per volta
    for token in vocabolario:
        if len(token) > 1:   #solo se token più lungo di 1 (semplice filtro)
            #calcolo frequenza del token e se questa è uguale a 1 aggiungo il token alla lista di hapax
            if corpus.count(token) == 1:
                hapax.append(token)
    return hapax

def percentualePOS(corpus):
    contatore_piene = 0
    contatore_funzionali = 0
    #lunghezzaTOT è un contatore che incrementa solo quando viene trovata una parola piena o una funzionale e viene usato per calcolare la percentuale
    #in questo modo evito di usare la lunghezza del corpus, che include elementi che non sono ne parole funzionali ne piene (come la punteggiatura)
    #così la somma delle due percentuali sarà 100, se usassi la lunghezza del corpus risulterebbe minore
    lunghezzaTOT = 0
    #eseguo POS tagging del corpus
    corpusPOS = nltk.pos_tag(corpus)
    #scorro il corpus POS taggato token per token
    for token in corpusPOS:
        #controllo se la POS di quel token fa parte di lista_POS_piene o lista_POS_funzionali e aumento i contatori di conseguenza
        if token[1] in lista_POS_piene:
            contatore_piene += 1
            lunghezzaTOT += 1
        elif token[1] in lista_POS_funzionali:
            contatore_funzionali += 1
            lunghezzaTOT += 1
    #calcolo la percentuale sia per le piene che per le vuote
    percentualePiene = (contatore_piene/lunghezzaTOT)*100
    percentualeFunzionali = (contatore_funzionali/lunghezzaTOT)*100
    return percentualePiene, percentualeFunzionali

def main(file1, file2):
    #apro i file e assegno il loro contenuto ad una variabile
    with open(file1, "r", encoding="utf-8") as fileInput1:
        raw1 = fileInput1.read()
    with open(file2, "r", encoding="utf-8") as fileInput2:
        raw2 = fileInput2.read()
    #carico il modello di tokenizzazione
    sentTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #estreggo le singole frasi
    frasi1 = sentTokenizer.tokenize(raw1)
    frasi2 = sentTokenizer.tokenize(raw2)
    #estraggo vocabolario, corpus e lunghezza corpus dei due testi
    vocab1, corpus1, lenCorpus1 = estraiTokens(frasi1)
    vocab2, corpus2, lenCorpus2 = estraiTokens(frasi2)
    #calcolo lunghezza media delle frasi in termini di token e dei token in termini di caratteri (punteggiatura esclusa) 
    mediaFrasi1, mediaCaratteri1 = calcoloMedia(frasi1)
    mediaFrasi2, mediaCaratteri2 = calcoloMedia(frasi2)
    print("lunghezza del corpus giornalistico: ", lenCorpus1, "numero di frasi: ", len(frasi1))
    print("lunghezza del corpus letterario: ", lenCorpus2, "numero di frasi: ", len(frasi2))
    print()
    print("la media delle frasi in termini di token del corpus giornalistico è: ", mediaFrasi1, "e la rispettiva media in caratteri dei token è: ", mediaCaratteri1)
    print("la media delle frasi in termini di token del corpus letterario è: ", mediaFrasi2, "e la rispettiva media in caratteri dei token è: ", mediaCaratteri2)
    #seleziono i primi 1000 token per calcolarne gli hapax
    corpus1mille = corpus1[0:999]
    #calcolo gli hapax
    HapaxMille1 = CalcoloHapax(sorted(list(set(corpus1mille))), corpus1mille)
    print("\nHapax estratti dai primi 1000 token del testo giornalistico: ")
    print(HapaxMille1)
    corpus2mille = corpus2[0:999]
    HapaxMille2 = CalcoloHapax(sorted(list(set(corpus2mille))), corpus2mille)
    print("\nHapax estratti dai primi 1000 token del testo letterario: ")
    print(HapaxMille2)
    print("\n")
    print("crescita del vocabolario e TTR all'incrementare del corpus giornalistico: ")
    #ciclo che scorre di 500 unità pr volta il corpus giornalistico
    for index in range(0, lenCorpus1, 500):
        tokens500 = corpus1[0:index+500]
        #creo vocabolario di tokens500
        vocabolario500 = list(set(tokens500))
        #calcolo type-token ratio
        ttr500 = len(vocabolario500) / len(tokens500)
        print("dimensioni del corpus: ", len(tokens500), "\tdimensioni del vocabolario: ", len(vocabolario500), "\tTTR: ", ttr500)
    print("\n")
    print("crescita del vocabolario e TTR all'incrementare del corpus letterario: ")
    #ciclo che scorre di 500 unità pr volta il corpus letterario
    for index in range(0, lenCorpus2, 500):
        tokens500 = corpus2[0:index+500]
        #creo vocabolario di token500
        vocabolario500 = list(set(tokens500))
        ttr500 = len(vocabolario500) / len(tokens500)
        print("dimensioni del corpus: ", len(tokens500), "\tdimensioni del vocabolario: ", len(vocabolario500), "\tTTR: ", ttr500)
    #calcolo percentuale parole piene e vuote
    print("\n\n")
    percentualePiene1, PercentualeVuote1 = percentualePOS(corpus1)
    percentualePiene2, PercentualeVuote2 = percentualePOS(corpus2)
    print("La percentuale di parole piene per corpus giornalistico: ", percentualePiene1, "\tPercentuale di parole vuote", PercentualeVuote1)
    print("La percentuale di parole piene per corpus letterario: ", percentualePiene2, "\tPercentuale di parole vuote", PercentualeVuote2)
    return

if __name__ == "__main__":
 main (sys.argv[1], sys.argv[2])
