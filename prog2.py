import sys
import string
import math
import nltk
from nltk.util import bigrams, trigrams

def main(file1, file2):
    #apro i file e assegno il loro contenuto ad una variabile
    #apro i file e assegno il loro contenuto ad una variabile
    with open(file1, "r", encoding="utf-8") as fileInput1:
        raw1 = fileInput1.read()
    with open(file2, "r", encoding="utf-8") as fileInput2:
        raw2 = fileInput2.read()
    #carico il modello di tokenizzazione
    sentTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #estreggo le singole frasi con tokenize
    frasi1 = sentTokenizer.tokenize(raw1)
    frasi2 = sentTokenizer.tokenize(raw2)
    #estraggo vocabolario, corpus e lunghezza corpus dei due testi
    vocab1, corpus1, lenCorpus1 = estraiTokens(frasi1)
    vocab2, corpus2, lenCorpus2 = estraiTokens(frasi2)
    #faccio POS tagging dei corpus
    corpusPOS1 = nltk.pos_tag(corpus1)
    corpusPOS2 = nltk.pos_tag(corpus2)
    print("Dati estratti dal corpus GIORNALISTICO:\n")
    EstraiFrequenze(corpusPOS1)
    print("\n")
    print("Dati estratti dal corpus LETTERARIO:\n")
    EstraiFrequenze(corpusPOS2)
    #creo bigrammi e trigrammi POS taggati
    bigrammi1POS = list(bigrams(corpusPOS1))
    bigrammi2POS = list(bigrams(corpusPOS2))
    #estraggo i dati richiesti dal punto 2
    print("\n")
    print("PER IL CORPUS GIORNALISTICO:")
    ventiBigrammi(bigrammi1POS, corpus1)
    print("\n")
    print("PER IL CORPUS LETTERARIO:")
    ventiBigrammi(bigrammi2POS, corpus2)
    print("\n")
    #estraggo i dati richiesti dal punto 3
    print("PER IL CORPUS GIORNALISTICO:")
    EstraiPuntoTre(corpus1, frasi1)
    print("PER IL CORPUS LETTERARIO:")
    EstraiPuntoTre(corpus2, frasi2)
    print("\n")
    #estraggo i dati richiesti dal punto 4
    print("PER IL CORPUS GIORNALISTICO:")
    EstraiNomi(corpusPOS1)
    print("\n")
    print("PER IL CORPUS LETTERARIO:")
    EstraiNomi(corpusPOS2)
    return

def estraiTokens(frasi):
    #variabili per lunghezza del corpus e per contenere il corpus
    lenCorpus = 0
    corpus = []
    #ciclo che scorre frase per frase
    for frase in frasi:
        #da frase estraggo i tokens 
        tokens = nltk.word_tokenize(frase)
        corpus += tokens
    #numero dei token del file
    lenCorpus = len(corpus)
    #creo lista dei token DIVERSI e ordinati
    vocabolario = list(sorted(set(corpus)))
    return vocabolario, corpus, lenCorpus
    
def CalcoloHapax (vocabolario, corpus):
    hapax = []
    #scorro vocabolario un token per volta
    for token in vocabolario:
        if len(token) > 1:   #solo se token più lungo di 1 (semplice filtro)
            #calcolo frequenza del token e se questa è uguale a 1 aggiungo il token alla lista di hapax
            if corpus.count(token) == 1:
                hapax.append(token)
    return hapax

#dato corpus POS taggato mi restituisce una lista con solo le POS
def EstraiSequenzaPOS(testoAnalizzatoPOS):
    listaPOS = []
    for token in testoAnalizzatoPOS:
        listaPOS.append(token[1])
    return listaPOS

#estrae i dati richiesti nel punto 1 del secondo programma
def EstraiFrequenze(testoAnalizzatoPOS):
    count = 1
    listaPOS = (EstraiSequenzaPOS(testoAnalizzatoPOS))
    #creo distribuzione frequenza della POS
    distFreq = nltk.FreqDist(listaPOS)
    #scorro la lista delle 10 pos più frequenti e le stampo
    for pos in distFreq.most_common(10):
        print(count,"° POS è: ", pos[0], "  con frequenza:", pos[1])
        count += 1
    #estraggi i bigrammi e i trigrammi di listaPOS
    bigrammiPOS = bigrams(listaPOS)
    trigrammiPOS = trigrams(listaPOS)
    print()
    count = 1
    #stampo i primi 10 bigrammi di POS dalla dist di freq dei bigrammi di POS
    for pos in nltk.FreqDist(bigrammiPOS).most_common(10):
        print(count,"° bigramma di POS è: ", pos[0], "  con frequenza:", pos[1])
        count += 1
    print()
    count = 1
    #stampo i primi 10 trigrammi di POS dalla dist di freq dei trigrammi di POS
    for pos in nltk.FreqDist(trigrammiPOS).most_common(10):
        print(count,"° trigramma di POS è: ", pos[0], "  con frequenza:", pos[1])
        count += 1
    print()
    #liste che contengono gli identificatori degli aggettivi e degli avverbi
    avverbi = ["RB", "RBR", "RBS"]    
    aggettivi = ["JJ", "JJS", "JJR"]
    #mi creo lista di aggettivi vuota
    listaAggettivi = []
    #scorro il testo con le POS
    for token in testoAnalizzatoPOS:
        #se la POS è uguale ad una contenuta nella lista aggettivi aggiungo il token alla lista di aggettivi
        if token[1] in aggettivi:
            listaAggettivi.append(token[0])
    #come sopra, ma per gli avverbi
    listaAvverbi = []
    for token in testoAnalizzatoPOS:
        if token[1] in avverbi:
            listaAvverbi.append(token[0])
    #creo dist freq per avverbi e aggettivi
    distAvverbi = nltk.FreqDist(listaAvverbi)
    distAggettivi = nltk.FreqDist(listaAggettivi)
    #stampo primi 20 aggettivi
    print("Lista dei primi 20 aggettivi più frequenti")
    for pos in distAggettivi.most_common(20):
        print(pos[0], "\tcon frequenza:\t", pos[1])
    #stampo primi 20 avverbi
    print()
    print("Lista dei primi 20 avverbi più frequenti")
    for pos in distAvverbi.most_common(20):
        print(pos[0], "\tcon frequenza:\t", pos[1]) 
    return

#estrae dati richiesti dal punto 2 del programma
def ventiBigrammi(bigrammiPOS, corpus):
    #lista che conterrà tutti i bigrammi aggettivo + sostantivo e i cui token hanno frequenza > 3
    listaBigrammi = []
    #liste con gli identificatori di sostantivi e aggettivi
    listaPOSsostantivi = ["NN", "NNS", "NNP", "NNPS"]
    listaPOSaggettivi = ["JJ", "JJR", "JJS"]
    #scorro i bigrammi
    for bigramma in bigrammiPOS:
        #se la pos del primo token è uguale ad uno degli elementi nella lista degli aggettivi
        # e se la pos del secondo token e uguale ad uno degli elementi nella lista sostantivi
        if bigramma[0][1] in listaPOSaggettivi and bigramma[1][1] in listaPOSsostantivi:
            #se i token compaiono più di tre volte nel corpus
            if corpus.count(bigramma[0][0]) > 3 and corpus.count(bigramma[1][0]) > 3:
                #allora aggiungo il bigramma alla lista
                listaBigrammi.append(bigramma)
    #creo distribuzione di frequenza dei bigrammi estratti dal ciclo precedente
    distBigrammi = nltk.FreqDist(listaBigrammi)
    #estraggo i 20 più frequenti
    mostFreqBigrammi = distBigrammi.most_common(20)
    print("20 bigrammi composti da Aggettivo e Sostantivo dove ogni token ha una frequenza maggiore di 3 ordinati per frequenza:")
    for bigramma in mostFreqBigrammi:
        #scorro mostFreqBigrammi per salvare i due token e la freq per poi successivamente stamparli
        tok1 = bigramma[0][0][0]
        tok2 = bigramma[0][1][0]
        freq = bigramma[1]
        print("bigramma:\t", tok1, tok2, "\tfrequenza: ", freq)
    print()
    #estrazione dei 20 bigrammi con probabilità condizionata massima, indicando anche la relativa probabilità
    print("20 bigrammi composti da Aggettivo e Sostantivo e dove ogni token ha una frequenza maggiore di 3 ordinati per probabilità condizionata massima:")
    #creo una lista vuota
    listaProbCondizionata = []
    #scorro la distribuzione di frequenza
    for bigramma in distBigrammi.most_common():
        #salvo il token 1 e 2 e la freq del bigramma in delle variabili
        tok1 = bigramma[0][0][0]
        tok2 = bigramma[0][1][0]
        freqBigramma = bigramma[1]
        #conto quante volte il token 1 compare nel corpus
        freqToken1 = corpus.count(tok1)
        #calcolo la sua probabilità condizionata
        probCondizionata = freqBigramma/freqToken1
        #aggiungo la tupla formata dal token 1 e il token 2 e la loro prob condizionata alla lista probCondizionata
        listaProbCondizionata.append((tok1, tok2, probCondizionata))
    #creo lista che conterrà i 20 bigrammi con prob condizionata più alta
    listaVentiCondizionata = []
    #ciclo while che cicla per 20 volte
    i = 0
    while (i < 20):
        i = i + 1
        #variabile per salvare la probabilità condizionata massima
        probCondizionataMax = 0.0
        #for che trova la probcondizionata max e salva il bigramma corrispondente in una variabile
        for bigramma in listaProbCondizionata:
            if bigramma[2] > probCondizionataMax:
                probCondizionataMax = bigramma[2]
                bigrammaMax = bigramma
        #aggiungo il bigramma alla lista dei venti bigrammi con prob condizionata più alta       
        listaVentiCondizionata.append(bigrammaMax)
        #rimuovo il bigramma con probabilità più alta dalla lista
        #così durante il primo ciclo WHILE il ciclo FOR troverà la prob condizionata max, al secondo ciclo la seconda max, al terzo la terza e così via
        listaProbCondizionata.remove(bigrammaMax)
    #scorro la lista dei venti bigrammi con prob condizionata più alta e salvo le singole parti in variabili che poi stampo
    for bigramma in listaVentiCondizionata:
        tok1 = bigramma[0]
        tok2 = bigramma[1]
        prob = bigramma[2]
        print("bigramma:\t", tok1, tok2, "\tprobabilità Condizionata: ", prob)
    #estrazione dei 20 bigrammi con local mutual information massima
    print()
    print("20 bigrammi composti da Aggettivo e Sostantivo e dove ogni token ha una frequenza maggiore di 3 ordinati per Local Mutual Information:")
    #creo una lista vuota
    listaLMI = []
    #scorro la distribuzione di frequenza
    for bigramma in distBigrammi.most_common():
        #salvo il token 1 e 2 e la freq del bigramma in delle variabili
        tok1 = bigramma[0][0][0]
        tok2 = bigramma[0][1][0]
        freqBigramma = bigramma[1]
        #conto quante volte token 1 e il token 2 compaiono nel corpus
        freqToken1 = corpus.count(tok1)
        freqToken2 = corpus.count(tok2)
        #calcolo la probabilità dei due token
        probToken1 = freqToken1/len(corpus)
        probToken2 = freqToken2/len(corpus)
        #calcolo la probabilità condizionata
        probCondizionata = freqBigramma/freqToken1
        #probabilità congiunta -> probabilità condizionata * prob del token1
        probCongiunta = probCondizionata * probToken1
        #calcolo local mutual information
        temp = probCongiunta/(probToken1*probToken2)
        Mi = math.log(temp, 2)
        lmi = Mi * freqBigramma
        #aggiungo la tupla alla lista listaLMI
        listaLMI.append((tok1, tok2, lmi))
    #lista che conterrà i 20 bigrammi con LMI più alta
    listaLMImax = []
    #ciclo while che cicla 20 volte
    i = 0
    while (i < 20):
        i = i + 1
        #variabile per trovare la LMI massima
        LMImax = 0.0
        #variabile per salvare temporaneamente il bigramma con LMI max
        bigrammaMax = ()
        #for che trova la LMI max e salva il bigramma corrispondente in una variabile
        for bigramma in listaLMI:
            if bigramma[2] > LMImax:
                LMImax = bigramma[2]
                bigrammaMax = bigramma
        #aggiungo il bigramma con probabilità più alta alla lista        
        listaLMImax.append(bigrammaMax)
        #rimuovo il bigramma con LMI più alta dalla lista cosi durante il prossimo ciclo while troverò il bigramma con LMI più alta successivo, e non lo stesso di prima
        listaLMI.remove(bigrammaMax)
    #scorro la lista dei venti bigrammi con LMI più alta e salvo le singole parti in variabili che poi stampo
    for bigramma in listaLMImax:
        tok1 = bigramma[0]
        tok2 = bigramma[1]
        localMutualInformation = bigramma[2]
        print("bigramma:\t", tok1, tok2, "\tLocal Mutual Information: ", localMutualInformation)
    return

def EstraiPuntoTre(corpus, frasi):
    #creo distribuzione di frequenza dei token, necessaria per catene di markov
    freqDist = nltk.FreqDist(corpus)
    #creo bigrammi e trigrammi del corpus
    bigrammi = list(bigrams(corpus))
    trigrammi = list(trigrams(corpus))
    #creo le loro distribuzioni di frequenza
    distFreqBigrammi = nltk.FreqDist(bigrammi)
    distFreqTrigrammi = nltk.FreqDist(trigrammi)
    #liste in cui salvo le frasi con numero di token tra 6 e 25 e in cui ogni token ha freq >= 2
    frasiFiltrate = []
    #ciclo che scorre frase per frase
    for frase in frasi:
        #elimino dalla frase la punteggiatura   https://www.w3schools.com/python/ref_string_maketrans.asp
        frase = frase.translate(str.maketrans("", "", string.punctuation))
        #da frase estraggo i tokens 
        tokens = nltk.word_tokenize(frase)
        #calcolo quanti token ci sono nella frase e se il numero e tra 6 e 25 passo ad un ulteriore ciclo
        if len(tokens) >= 6 and len(tokens) <=25:
            #ciclo che scorre la frase token per token
            for token in tokens:
                #controllo se i token hano una frequenza minore di 2 nel corpus, se è cosi interrompo il ciclo
                if corpus.count(token) < 2:
                    break
            #se invece il ciclo scorre tutti i token senza venire interrotto aggiungo la frase alla lista frasiFiltrate    
            else:
                frasiFiltrate.append(frase)
    #variabili in cui salvo la frase con media più alta e quella con media più bassa
    fraseMax = ""
    fraseMin = ""
    #inizializzo i valori massimi e minimi della media
    #massima inizia da 0, minima inizia da un numero molto alto per essere sicuro che verrà sostituito
    mediaMax = 0.0
    mediaMin = 1000.0
    for frase in frasiFiltrate:
        #lista in cui salvo i valori di frequenza dei singoli token delle frasi
        listaFreq = []
        #ciclo che inserisce in una lista i valori di frequenza dei singoli token della frase
        for token in frase:
            listaFreq.append(freqDist[token])
        #calcolo la media della distribuzione di frequenza dei token
        media = sum(listaFreq) / len(listaFreq)
        if media > mediaMax:
            mediaMax = media
            fraseMax = frase
        if media < mediaMin:
            mediaMin = media
            fraseMin = frase      
    print("La frase con media di distribuzione più alta è: ", fraseMax, "\ncon media di: ", mediaMax)
    print("La frase con media di distribuzione più bassa è: ", fraseMin, "\ncon media di: ", mediaMin)
    #variabili per salvare la probabilità massima e la rispettiva frase
    fraseMax = []
    probMax = 0.0
    #calcolo la probabilità delle frasi usando Markov ordine 2
    for frase in frasiFiltrate:
        #creo tokens, bigrammi e trigrammi della frase
        tokens = nltk.word_tokenize(frase)
        bigrammiFrase = list(bigrams(tokens))
        trigrammiFrase = list(trigrams(tokens))
        #calcolo prob della frase
        prob = CatenaMarkov2(bigrammiFrase, trigrammiFrase, distFreqBigrammi, distFreqTrigrammi, freqDist, corpus)
        #se la prob è maggiore della probMax salvo il valore e la frase corrispondente
        if prob > probMax:
            probMax = prob
            fraseMax = frase
    print("La frase con probabilità Markov di ordine 2 maggiore è: ", fraseMax, "\ncon valore di probabilità", probMax)
    print()
    return

def CatenaMarkov2 (bigrammiFrase, trigrammiFrase, distFreqBigrammi, distFreqTrigrammi, distFreq, corpus):
    #i calcoli delle probabilità sono effettuati usando lo ADD-ONE SMOOTHING, per evitare casi con bigrammi o trigrammi con frequenza zero
    vocabolario = list(sorted(set(corpus)))
    #prob è inizializzato con la freq del primo token
    prob = (distFreq[bigrammiFrase[0][0]] + 1) / (len(corpus) + len(vocabolario))
    #calcolo la probabilità del primo bigramma e aggiorno la probabailità
    probBigramma1 = (distFreqBigrammi[bigrammiFrase[0]] + 1) / (distFreq[bigrammiFrase[0][0]] + len(vocabolario))
    prob = prob * probBigramma1
    #unisco i trigrammi e i bigrammi in un'unica tupla così posso scorrerli in parallelo
    trigrammiBigrammi = zip(trigrammiFrase, bigrammiFrase)
    #lo converto in lista
    listaTrigrammiBigrammi = list(trigrammiBigrammi)
    #ciclo che scorre la lista dei trigrammi e bigrammi ed esegue i calcoli per la probabilità
    for trigrammaBigramma in listaTrigrammiBigrammi:
        prob = prob * (distFreqTrigrammi[trigrammaBigramma[0]] + 1) / ((distFreqBigrammi[trigrammaBigramma[1]]) + len(vocabolario))
    return prob

def EstraiNomi (pos):
    #creo albero delle named entity
    alberoNE = nltk.ne_chunk(pos)
    nomi = []
    #scorro ogni nodo all'interno dell'albero sintattico
    for nodo in alberoNE:
        NE = ""
        if hasattr(nodo, "label"):
            if nodo.label() == "PERSON":
                #scorro le foglie, cosi da trovare nomi composti da più di 1 token
                for partNE in nodo.leaves():
                    NE = NE + " " + partNE[0]
                nomi.append(NE)
    #creo distribuzione di frequenza dei nomi
    DistNomiPropri = nltk.FreqDist(nomi)
    #salvo i 15 nomi più frequenti
    nomiFrequenti = DistNomiPropri.most_common(15)
    print("nomi propri di persona in ordine di frequenza\n")
    for nome in nomiFrequenti:
        print("nome proprio:\t", nome[0], "\t\tfrequenza: ", nome[1])
    return

main (sys.argv[1], sys.argv[2])