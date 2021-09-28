from nltk.corpus import wordnet as wn

'''
    Custom featurizations based on datasets used. Code will dynamically "use get_characteristics_<DATASET>",
    where '-' in <DATASET> is replaced with "_".
'''
# DATASET: amazon
def get_characteristics_amazon(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.0786094674556213
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.11734437059252655
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.17727608008429926
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.2108954770203453
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.17762462511145335
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.1588591229634433
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.07939085677231093
    	X.append(1)
    else:
    	X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if isLocal and -1 < len(word) <= 3: # 0.10841452541136419
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if isLocal and 3 < len(word) <= 4: # 0.25564399773040447
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if isLocal and 4 < len(word) <= 5: # 0.2222783091513334
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if isLocal and 5 < len(word) <= 6: # 0.19031328523952337
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if isLocal and 6 < len(word) <= 8: # 0.16582130988084623
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if isLocal and 8 < len(word) <= 10000000: # 0.057528572586528326
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.07191861878900867
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.10791784874766962
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.12556071168031127
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.22060894058523142
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.3054908000324228
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.16850308016535626
    	X.append(1)
    else:
    	X.append(0)

    tfAttrFeatures = [0]*len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    #print("featurization_util X: ", X)


    #'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]: # 0.0786094674556213
        	X.append(1)
        else:
        	X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        #self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]: # 0.07191861878900867
        	X.append(1)
        else:
        	X.append(0)

    #How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0
    if self.first:
        self.end = len(X)
    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)

    if self.first:
        for idx, i in enumerate(self.FEATURES):
            print(str(idx)+ ": "+ str(i))
        self.first=False
        print(self.start)
        print(self.end)

    return X
def get_characteristics_amazon_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.06589386282381098
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.0806778302698074
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.17854751179240969
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.20664528358503892
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.1994286460105854
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.182026989279184
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.08677987623916361
    	X.append(1)
    else:
    	X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3: # 0.1081641776662068
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4: # 0.25576202691609123
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5: # 0.22148261725015767
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6: # 0.1910471001101479
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8: # 0.16616380440762277
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000: # 0.05738027364977367
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    # Term Frequency normalized single attribute
    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.04521399548918221
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.0838572371302026
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.1055022706563862
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.19715920942368143
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.31638910133726594
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.2518781859632816
    	X.append(1)
    else:
    	X.append(0)

    tfAttrFeatures = [0] * len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    
    return X

# DATASET: google
def get_characteristics_google(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.0786094674556213
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.11734437059252655
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.17727608008429926
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.2108954770203453
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.17762462511145335
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.1588591229634433
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.07939085677231093
    	X.append(1)
    else:
    	X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if isLocal and -1 < len(word) <= 3: # 0.10841452541136419
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if isLocal and 3 < len(word) <= 4: # 0.25564399773040447
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if isLocal and 4 < len(word) <= 5: # 0.2222783091513334
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if isLocal and 5 < len(word) <= 6: # 0.19031328523952337
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if isLocal and 6 < len(word) <= 8: # 0.16582130988084623
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if isLocal and 8 < len(word) <= 10000000: # 0.057528572586528326
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.07191861878900867
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.10791784874766962
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.12556071168031127
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.22060894058523142
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.3054908000324228
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.16850308016535626
    	X.append(1)
    else:
    	X.append(0)

    tfAttrFeatures = [0]*len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    #print("featurization_util X: ", X)


    #'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]: # 0.0786094674556213
        	X.append(1)
        else:
        	X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        #self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]: # 0.07191861878900867
        	X.append(1)
        else:
        	X.append(0)
    if self.first:
        self.end = len(X)
    #How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0

    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)

    if self.first:
        self.first = False
    
    return X
def get_characteristics_google_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.06589386282381098
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.0806778302698074
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.17854751179240969
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.20664528358503892
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.1994286460105854
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.182026989279184
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.08677987623916361
    	X.append(1)
    else:
    	X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3: # 0.1081641776662068
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4: # 0.25576202691609123
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5: # 0.22148261725015767
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6: # 0.1910471001101479
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8: # 0.16616380440762277
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000: # 0.05738027364977367
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    # Term Frequency normalized single attribute
    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.04521399548918221
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.0838572371302026
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.1055022706563862
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.19715920942368143
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.31638910133726594
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.2518781859632816
    	X.append(1)
    else:
    	X.append(0)

    tfAttrFeatures = [0] * len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    
    return X

# DATASET: drug_reviews
def get_characteristics_drug_reviews(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    if 0 < idf <= 0.165:  # 0.1455035726255316
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.165 < IDF <= 0.22')
    if 0.165 < idf <= 0.22:  # 0.1403783968254211
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.22 < IDF <= 0.285')
    if 0.22 < idf <= 0.285:  # 0.1420801571985841
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.285 < IDF <= 0.36')
    if 0.285 < idf <= 0.36:  # 0.14831943511349915
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.36 < IDF <= 0.45')
    if 0.36 < idf <= 0.45:  # 0.14772106433480606
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.45 < IDF <= 0.58')
    if 0.45 < idf <= 0.58:  # 0.142784122820576
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.58 < IDF <= 1')
    if 0.58 < idf <= 1:  # 0.13321325108158197
        X.append(1)
    else:
        X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('kewords 0-3')  # 0.16313625335587897
    if len(word) <= 3:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 4')  # 0.2786001585566414
    if len(word) == 4:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 5')  # 0.21846818742560653
    if len(word) == 5:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 6')  # 0.143921908238893
    if len(word) == 6:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 7-8')  # 0.14205763481610942
    if 7 <= len(word) <= 8:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 9+')  # 0.05381585760687069
    if 9 <= len(word):
        X.append(1)
    else:
        X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.3')
    if 0 < normTF <= 0.3:  # 0.2960206047676837
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.3 < normTF <= 0.4')
    if 0.3 < normTF <= 0.4:  # 0.36868516055772443
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.4 < normTF <= 0.6')
    if 0.4 < normTF <= 0.6:  # 0.2310767154188366
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.6 < normTF <= 1')
    if 0.6 < normTF <= 1:  # 0.1042175192557553
        X.append(1)
    else:
        X.append(0)

    tfAttrFeatures = [0] * len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+')  # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    # print("featurization_util X: ", X)

    # 'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]:  # 0.0786094674556213
            X.append(1)
        else:
            X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        # self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]:  # 0.07191861878900867
            X.append(1)
        else:
            X.append(0)
    if self.first:
        self.end = len(X)

    # How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0

    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word] / max(
                list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word] / max(
                list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)

    if self.first:
        self.first = False

    return X
def get_characteristics_drug_reviews_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []
    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.165')
    if 0 < idf <= 0.165:  # 0.1455035726255316
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.165 < IDF <= 0.22')
    if 0.165 < idf <= 0.22:  # 0.1403783968254211
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.22 < IDF <= 0.285')
    if 0.22 < idf <= 0.285:  # 0.1420801571985841
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.285 < IDF <= 0.36')
    if 0.285 < idf <= 0.36:  # 0.14831943511349915
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.36 < IDF <= 0.45')
    if 0.36 < idf <= 0.45:  # 0.14772106433480606
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.45 < IDF <= 0.58')
    if 0.45 < idf <= 0.58:  # 0.142784122820576
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.58 < IDF <= 1')
    if 0.58 < idf <= 1:  # 0.13321325108158197
        X.append(1)
    else:
        X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('kewords 0-3')  # 0.16313625335587897
    if len(word) <= 3:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 4')  # 0.2786001585566414
    if len(word) == 4:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 5')  # 0.21846818742560653
    if len(word) == 5:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 6')  # 0.143921908238893
    if len(word) == 6:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 7-8')  # 0.14205763481610942
    if 7 <= len(word) <= 8:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 9+')  # 0.05381585760687069
    if 9 <= len(word):
        X.append(1)
    else:
        X.append(0)
    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    # Term Frequency normalized single attribute
    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    self.FEATURES.append('0 < normTF <= 0.3')
    if 0 < normTF <= 0.3:  # 0.2960206047676837
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.3 < normTF <= 0.4')
    if 0.3 < normTF <= 0.4:  # 0.36868516055772443
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.4 < normTF <= 0.6')
    if 0.4 < normTF <= 0.6:  # 0.2310767154188366
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.6 < normTF <= 1')
    if 0.6 < normTF <= 1:  # 0.1042175192557553
        X.append(1)
    else:
        X.append(0)

    tfAttrFeatures = [0] * len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+')  # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)

    return X

# DATASET: wikipedia
def get_characteristics_wikipedia(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    if 0 < idf <= 0.11: # 0.14346267726351877
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.11 < IDF <= 0.23')
    if 0.11 < idf <= 0.23: # 0.14841826398628644
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.23 < IDF <= 0.36')
    if 0.23 < idf <= 0.36: # 0.1398472806607449
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.36 < IDF <= 0.51')
    if 0.36 < idf <= 0.51: # 0.1423094904160823
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.51 < IDF <= 0.67')
    if 0.51 < idf <= 0.67: # 0.1427146641732897
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.67 < IDF <= 0.88')
    if 0.67 < idf <= 0.88: # 0.11821723546828736
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.88 < IDF <= 1')
    if 0.88 < idf <= 1: # 0.16503038803179054
        X.append(1)
    else:
        X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3: # 0.08131525635031947
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4: # 0.20034283933302166
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5: # 0.21873149446781986
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6: # 0.17768427614149915
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8: # 0.19678977715443355
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000: # 0.12513635655290634
        X.append(1)
    else:
        X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.15583606046439147
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.19')
    if 0.15 < normTF <= 0.19: # 0.12454417952314166
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.19 < normTF <= 0.2')
    if 0.19 < normTF <= 0.2: # 0.20455041296556023
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.2476858345021038
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.45')
    if 0.25 < normTF <= 0.45: # 0.13429951690821257
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.45 < normTF <= 1')
    if 0.45 < normTF <= 1: # 0.13308399563659032
        X.append(1)
    else:
        X.append(0)

    tfAttrFeatures = [0]*len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]


    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    #print("featurization_util X: ", X)


    #'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]: # 0.0786094674556213
        	X.append(1)
        else:
        	X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        #self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]: # 0.07191861878900867
        	X.append(1)
        else:
        	X.append(0)
    if self.first:
        self.end = len(X)
    #How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0

    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)

    if self.first:
        self.first = False
    
    return X
def get_characteristics_wikipedia_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []
    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.11')
    if 0 < idf <= 0.11: # 0.14346267726351877
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.11 < IDF <= 0.23')
    if 0.11 < idf <= 0.23: # 0.14841826398628644
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.23 < IDF <= 0.36')
    if 0.23 < idf <= 0.36: # 0.1398472806607449
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.36 < IDF <= 0.51')
    if 0.36 < idf <= 0.51: # 0.1423094904160823
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.51 < IDF <= 0.67')
    if 0.51 < idf <= 0.67: # 0.1427146641732897
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.67 < IDF <= 0.88')
    if 0.67 < idf <= 0.88: # 0.11821723546828736
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.88 < IDF <= 1')
    if 0.88 < idf <= 1: # 0.16503038803179054
        X.append(1)
    else:
        X.append(0)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3: # 0.08131525635031947
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4: # 0.20034283933302166
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5: # 0.21873149446781986
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6: # 0.17768427614149915
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8: # 0.19678977715443355
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000: # 0.12513635655290634
        X.append(1)
    else:
        X.append(0)

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    # Term Frequency normalized single attribute
    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.15583606046439147
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.19')
    if 0.15 < normTF <= 0.19: # 0.12454417952314166
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.19 < normTF <= 0.2')
    if 0.19 < normTF <= 0.2: # 0.20455041296556023
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.2476858345021038
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.45')
    if 0.25 < normTF <= 0.45: # 0.13429951690821257
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.45 < normTF <= 1')
    if 0.45 < normTF <= 1: # 0.13308399563659032
        X.append(1)
    else:
        X.append(0)

    tfAttrFeatures = [0]*len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+')  # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)

    return X

# DATASET: wdc_1
def get_characteristics_wdc_1(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.26') # 0.14648216784175722
    if 0 < idf <= 0.26:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.26 < IDF <= 0.33') # 0.15976631646096373
    if 0.26 < idf <= 0.33:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.33 < IDF <= 0.38') # 0.14430366680509046
    if 0.33 < idf <= 0.38:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.38 < IDF <= 0.45')  # 0.15457574836844568
    if 0.38 < idf <= 0.45:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.45 < IDF <= 0.55')  # 0.1529552614751916
    if 0.45 < idf <= 0.55:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.55 < IDF <= 0.74')  # 0.1506081036869748
    if 0.55 < idf <= 0.74:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.74 < IDF <= 1')  # 0.09130873536157653
    if 0.74 < idf <= 1:
        X.append(1)
    else:
        X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('kewords 0-3') # 0.19023833725606046
    if len(word) <= 3:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 4') # 0.1797822665091961
    if len(word) == 4:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 5') # 0.18190332491345274
    if len(word) == 5:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 6') # 0.1675836972626426
    if len(word) == 6:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 7-8') # 0.1623419490803456
    if 7 <= len(word) <= 8:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 9+') # 0.11814934522957293
    if 9 <= len(word):
        X.append(1)
    else:
        X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15:  # 0.1532434774818385
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2:  # 0.19555700085183386
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25:  # 0.16835721784426141
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34:  # 0.18442079504164383
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5:  # 0.1568928505845778
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1:  # 0.14152865819584456
        X.append(1)
    else:
        X.append(0)

    tfAttrFeatures = [0]*len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    #print("featurization_util X: ", X)


    #'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]: # 0.0786094674556213
        	X.append(1)
        else:
        	X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        #self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]: # 0.07191861878900867
        	X.append(1)
        else:
        	X.append(0)
    if self.first:
        self.end = len(X)
    #How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0

    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)

    if self.first:
        self.first = False
    
    return X
def get_characteristics_wdc_1_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []
    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.26') # 0.14648216784175722
    if 0 < idf <= 0.26:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.26 < IDF <= 0.33') # 0.15976631646096373
    if 0.26 < idf <= 0.33:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.33 < IDF <= 0.38') # 0.14430366680509046
    if 0.33 < idf <= 0.38:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.38 < IDF <= 0.45')  # 0.15457574836844568
    if 0.38 < idf <= 0.45:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.45 < IDF <= 0.55')  # 0.1529552614751916
    if 0.45 < idf <= 0.55:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.55 < IDF <= 0.74')  # 0.1506081036869748
    if 0.55 < idf <= 0.74:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.74 < IDF <= 1')  # 0.09130873536157653
    if 0.74 < idf <= 1:
        X.append(1)
    else:
        X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('kewords 0-3') # 0.19023833725606046
    if len(word) <= 3:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 4') # 0.1797822665091961
    if len(word) == 4:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 5') # 0.18190332491345274
    if len(word) == 5:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 6') # 0.1675836972626426
    if len(word) == 6:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 7-8') # 0.1623419490803456
    if 7 <= len(word) <= 8:
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('kewords 9+') # 0.11814934522957293
    if 9 <= len(word):
        X.append(1)
    else:
        X.append(0)
    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    # Term Frequency normalized single attribute
    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15:  # 0.1532434774818385
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2:  # 0.19555700085183386
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25:  # 0.16835721784426141
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34:  # 0.18442079504164383
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5:  # 0.1568928505845778
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1:  # 0.14152865819584456
        X.append(1)
    else:
        X.append(0)

    tfAttrFeatures = [0]*len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)

    return X

# DATASET: wdc_2
def get_characteristics_wdc_2(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.15141129105847742
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.15251588683391187
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.14462037675795436
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.15060599921888973
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.14505929502972964
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.1482028988681201
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.10758425223291686
        X.append(1)
    else:
        X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3: # 0.1916996083498498
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4: # 0.1741164146308524
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5: # 0.17594828246991448
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6: # 0.16898764093747834
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8: # 0.1635463318842643
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000: # 0.12570172172764071
        X.append(1)
    else:
        X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.15655375243184447
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.21072502728410877
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.1689228525855115
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.17722078045326295
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.16265252638072192
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.12392506086455037
        X.append(1)
    else:
        X.append(0)

    tfAttrFeatures = [0]*len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]


    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    #print("featurization_util X: ", X)


    #'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]: # 0.0786094674556213
        	X.append(1)
        else:
        	X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        #self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]: # 0.07191861878900867
        	X.append(1)
        else:
        	X.append(0)
    if self.first:
        self.end = len(X)
    #How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0

    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)

    if self.first:
        self.first = False

    return X
def get_characteristics_wdc_2_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []
    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.15141129105847742
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.15251588683391187
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.14462037675795436
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.15060599921888973
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.14505929502972964
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.1482028988681201
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.10758425223291686
        X.append(1)
    else:
        X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3: # 0.1916996083498498
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4: # 0.1741164146308524
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5: # 0.17594828246991448
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6: # 0.16898764093747834
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8: # 0.1635463318842643
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000: # 0.12570172172764071
        X.append(1)
    else:
        X.append(0)
    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    # Term Frequency normalized single attribute
    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.15655375243184447
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.21072502728410877
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.1689228525855115
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.17722078045326295
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.16265252638072192
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.12392506086455037
        X.append(1)
    else:
        X.append(0)

    tfAttrFeatures = [0]*len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(header))
        for attrIndex, attr in enumerate(originAttr):
            if header == attr:
                tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)

    return X

# DATASET: cord
def get_characteristics_cord_1(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.0786094674556213
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.11734437059252655
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.17727608008429926
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.2108954770203453
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.17762462511145335
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.1588591229634433
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.07939085677231093
    	X.append(1)
    else:
    	X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if isLocal and -1 < len(word) <= 3: # 0.10841452541136419
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if isLocal and 3 < len(word) <= 4: # 0.25564399773040447
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if isLocal and 4 < len(word) <= 5: # 0.2222783091513334
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if isLocal and 5 < len(word) <= 6: # 0.19031328523952337
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if isLocal and 6 < len(word) <= 8: # 0.16582130988084623
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if isLocal and 8 < len(word) <= 10000000: # 0.057528572586528326
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.07191861878900867
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.10791784874766962
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.12556071168031127
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.22060894058523142
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.3054908000324228
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.16850308016535626
    	X.append(1)
    else:
    	X.append(0)

    for feat in self.lin_attribute_headers:  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(feat))
        if isLocal and feat in originAttr:
            X.append(originCounts[originAttr.index(feat)] / sum(originCounts))
        else:
            X.append(0)

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    #print("featurization_util X: ", X)


    #'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]: # 0.0786094674556213
        	X.append(1)
        else:
        	X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        #self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]: # 0.07191861878900867
        	X.append(1)
        else:
        	X.append(0)
    if self.first:
        self.end = len(X)
    #How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0

    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)
    """
    if isExternal and word in self.total_local_relevant:
        count_of_external_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
    if isExternal and word in self.total_external_relevant:
        count_of_external_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))
    self.FEATURES.append('How often external term appears in relevant local tuples')
    X.append(count_of_external_term_in_local_matches)
    self.FEATURES.append('How often external term appears in relevant external tuples')
    X.append(count_of_external_term_in_external_matches)
    """
    if self.first:
        self.first = False
    
    return X
def get_characteristics_cord_1_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.3369418439557301
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.18599553279422726
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.1506471808202407
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.1090283019516514
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.07788586007680175
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.07631585273889056
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.06318542766245819
    	X.append(1)
    else:
    	X.append(0)

    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3: # 0.10899589725943447
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4: # 0.13607903837236662
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5: # 0.17294712662120346
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6: # 0.2262426422190488
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8: # 0.2316955636969182
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000: # 0.12403973183102844
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    # Term Frequency normalized single attribute
    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.4415131432336879
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.22940923750051398
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.10959455205308029
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.09279630992881083
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.07201894676079851
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.054667810523108505
    	X.append(1)
    else:
    	X.append(0)

    for feat in self.lin_attribute_headers:  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(feat))
        if feat in originAttr:
            X.append(originCounts[originAttr.index(feat)] / sum(originCounts))
        else:
            X.append(0)

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    
    return X

# DATASET: cord_2
def get_characteristics_cord_2(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.0786094674556213
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.11734437059252655
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.17727608008429926
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.2108954770203453
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.17762462511145335
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.1588591229634433
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.07939085677231093
    	X.append(1)
    else:
    	X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if isLocal and -1 < len(word) <= 3: # 0.10841452541136419
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if isLocal and 3 < len(word) <= 4: # 0.25564399773040447
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if isLocal and 4 < len(word) <= 5: # 0.2222783091513334
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if isLocal and 5 < len(word) <= 6: # 0.19031328523952337
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if isLocal and 6 < len(word) <= 8: # 0.16582130988084623
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if isLocal and 8 < len(word) <= 10000000: # 0.057528572586528326
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.07191861878900867
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.10791784874766962
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.12556071168031127
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.22060894058523142
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.3054908000324228
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.16850308016535626
    	X.append(1)
    else:
    	X.append(0)

    for feat in self.lin_attribute_headers:  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(feat))
        if isLocal and feat in originAttr:
            X.append(originCounts[originAttr.index(feat)] / sum(originCounts))
        else:
            X.append(0)

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    #print("featurization_util X: ", X)


    #'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]: # 0.0786094674556213
        	X.append(1)
        else:
        	X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        #self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]: # 0.07191861878900867
        	X.append(1)
        else:
        	X.append(0)

    if self.first:
        self.end = len(X)
    #How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0

    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)
    """
    if isExternal and word in self.total_local_relevant:
        count_of_external_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
    if isExternal and word in self.total_external_relevant:
        count_of_external_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))
    self.FEATURES.append('How often external term appears in relevant local tuples')
    X.append(count_of_external_term_in_local_matches)
    self.FEATURES.append('How often external term appears in relevant external tuples')
    X.append(count_of_external_term_in_external_matches)
    """
    if self.first:
        self.first = False
    
    return X
def get_characteristics_cord_2_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.3369418439557301
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.18599553279422726
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.1506471808202407
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.1090283019516514
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.07788586007680175
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.07631585273889056
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.06318542766245819
        X.append(1)
    else:
        X.append(0)

    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3: # 0.1363376036851484
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4: # 0.11238321328309167
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5: # 0.13691423884159348
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6: # 0.16833925277387737
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8: # 0.20879675384906043
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000: # 0.2372289375672286
        X.append(1)
    else:
        X.append(0)
    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    # Term Frequency normalized single attribute
    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.008005366896133993
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.012941359088358686
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.024569475813840003
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.09584010959746855
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.4207764530636309
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.4378672355405679
        X.append(1)
    else:
        X.append(0)

    for feat in self.lin_attribute_headers:  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(feat))
        if feat in originAttr:
            X.append(originCounts[originAttr.index(feat)] / sum(originCounts))
        else:
            X.append(0)

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    
    return X

# DATASET: omdb-imdb plot
def get_characteristics_imdb(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.0786094674556213
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.11734437059252655
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.17727608008429926
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.2108954770203453
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.17762462511145335
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.1588591229634433
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.07939085677231093
    	X.append(1)
    else:
    	X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if isLocal and -1 < len(word) <= 3: # 0.10841452541136419
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if isLocal and 3 < len(word) <= 4: # 0.25564399773040447
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if isLocal and 4 < len(word) <= 5: # 0.2222783091513334
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if isLocal and 5 < len(word) <= 6: # 0.19031328523952337
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if isLocal and 6 < len(word) <= 8: # 0.16582130988084623
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if isLocal and 8 < len(word) <= 10000000: # 0.057528572586528326
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.07191861878900867
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.10791784874766962
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.12556071168031127
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.22060894058523142
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.3054908000324228
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.16850308016535626
    	X.append(1)
    else:
    	X.append(0)

    for feat in self.lin_attribute_headers:  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(feat))
        if isLocal and feat in originAttr:
            X.append(originCounts[originAttr.index(feat)] / sum(originCounts))
        else:
            X.append(0)

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    #print("featurization_util X: ", X)


    #'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]: # 0.0786094674556213
        	X.append(1)
        else:
        	X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        #self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]: # 0.07191861878900867
        	X.append(1)
        else:
        	X.append(0)
    if(self.start):
        end = len(X)
    #How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0

    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)

    if(self.first):
        self.first=False
    
    return X
def get_characteristics_imdb_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21:  # 0.065901620145161
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27:  # 0.08072331634156205
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34:  # 0.1789875815913057
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42:  # 0.20669248472479004
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52:  # 0.19925754275442367
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71:  # 0.18212527471589054
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1:  # 0.08631217972686699
        X.append(1)
    else:
        X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3:  # 0.10805918674946825
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4:  # 0.2557921363700965
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5:  # 0.22150869117816596
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6:  # 0.19106959103244378
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8:  # 0.16618336595663957
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000:  # 0.057387028713185934
        X.append(1)
    else:
        X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15:  # 0.045329723115660955
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2:  # 0.08391560191067456
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25:  # 0.10563302531808486
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34:  # 0.19701010295708057
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5:  # 0.31634064709430953
        X.append(1)
    else:
        X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1:  # 0.2517708996041895
        X.append(1)
    else:
        X.append(0)

    for feat in self.lin_attribute_headers:  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(feat))
        if feat in originAttr:
            X.append(originCounts[originAttr.index(feat)] / sum(originCounts))
        else:
            X.append(0)

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+')  # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)

    return X

# DATASET: omdb-imdb plot swapped
def get_characteristics_omdb(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    isLocal = False
    if word in self.local_term_binary:
        isLocal = self.local_term_binary[word]

    # [0] = idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.0786094674556213
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.11734437059252655
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.17727608008429926
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.2108954770203453
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.17762462511145335
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.1588591229634433
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.07939085677231093
    	X.append(1)
    else:
    	X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    attribute_count = 0
    if isLocal:
        attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if isLocal and -1 < len(word) <= 3: # 0.10841452541136419
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if isLocal and 3 < len(word) <= 4: # 0.25564399773040447
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if isLocal and 4 < len(word) <= 5: # 0.2222783091513334
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if isLocal and 5 < len(word) <= 6: # 0.19031328523952337
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if isLocal and 6 < len(word) <= 8: # 0.16582130988084623
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if isLocal and 8 < len(word) <= 10000000: # 0.057528572586528326
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if isLocal:
        normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.07191861878900867
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.10791784874766962
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.12556071168031127
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.22060894058523142
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.3054908000324228
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.16850308016535626
    	X.append(1)
    else:
    	X.append(0)

    for feat in self.lin_attribute_headers:  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(feat))
        if isLocal and feat in originAttr:
            X.append(originCounts[originAttr.index(feat)] / sum(originCounts))
        else:
            X.append(0)

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    if isLocal:
        for c in word:
            if not c.isalpha():
                encode = 1
                break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    #print("featurization_util X: ", X)


    #'self.FEATURES.append(\'{} < normTF <= {}\')'.format(ranges[0], ranges[1])
    self.FEATURES.append('Is local term')
    if isLocal:
        X.append(1)
    else:
        X.append(0)

    ### EXTERNAL FEATURES ###
    if self.first:
        self.start = len(X)
    isExternal = False
    if word in self.external_term_binary:
        isExternal = self.external_term_binary[word]

    self.FEATURES.append('Is external term')
    if isExternal:
        X.append(1)
    else:
        X.append(0)

    idf_ex = 0
    if word in self.idf_external:
        idf_ex = self.idf_external[word]
    for bucket in self.idf_external_bucket:
        self.FEATURES.append('{} < IDF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf_ex <= bucket[1]: # 0.0786094674556213
        	X.append(1)
        else:
        	X.append(0)

    externalTupleID = ''
    if self.computedExternalStats and isExternal and tupleID in self.matched_tuples:
        for externalTuple in self.matched_tuples[tupleID]:
            for sig in self.signalIndex_external[externalTuple]:
                if sig == signal:
                    externalTupleID = externalTuple
        #self.maximumTF_external[externalTupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex_external[externalTupleID]]) #should be calculated already

    self.FEATURES.append('maximum_TF_external_normalized')
    normTF_external = 0
    if self.computedExternalStats and isExternal and externalTupleID != '':
        normTF_external = (signal.getTermFrequncy() / self.maximumTF_external[externalTupleID])
    X.append(normTF_external)

    # Term Frequency normalized single attribute
    for bucket in self.tf_external_bucket:
        self.FEATURES.append('{} < normTF External <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF_external <= bucket[1]: # 0.07191861878900867
        	X.append(1)
        else:
        	X.append(0)
    if(self.start):
        end = len(X)
    #How often term appears in relevant answer
    count_of_local_term_in_external_matches = 0
    count_of_local_term_in_local_matches = 0
    count_of_external_term_in_local_matches = 0
    count_of_external_term_in_external_matches = 0

    if tupleID in self.matched_tuples:

        if isLocal and word in self.total_local_relevant:
            count_of_local_term_in_local_matches = self.total_local_relevant[word]/max(list(self.total_local_relevant.values()))
        if isLocal and word in self.total_external_relevant:
            count_of_local_term_in_external_matches = self.total_external_relevant[word]/max(list(self.total_external_relevant.values()))

    self.FEATURES.append('How often local term appears in relevant local tuples')
    X.append(count_of_local_term_in_local_matches)
    self.FEATURES.append('How often local term appears in relevant external tuples')
    X.append(count_of_local_term_in_external_matches)

    if(self.first):
        self.first=False
    
    return X
def get_characteristics_omdb_without_externalFeatures(self, signal, tupleID):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()

    self.FEATURES = []
    X = []

    # [0] = idf of term
    idf = self.idf[word]
    self.FEATURES.append('0 < IDF <= 0.21')
    if 0 < idf <= 0.21: # 0.065901620145161
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.21 < IDF <= 0.27')
    if 0.21 < idf <= 0.27: # 0.08072331634156205
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.27 < IDF <= 0.34')
    if 0.27 < idf <= 0.34: # 0.1789875815913057
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < IDF <= 0.42')
    if 0.34 < idf <= 0.42: # 0.20669248472479004
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.42 < IDF <= 0.52')
    if 0.42 < idf <= 0.52: # 0.19925754275442367
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.52 < IDF <= 0.71')
    if 0.52 < idf <= 0.71: # 0.18212527471589054
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.71 < IDF <= 1')
    if 0.71 < idf <= 1: # 0.08631217972686699
    	X.append(1)
    else:
    	X.append(0)
    idf_bucket_count = len(X)

    # [1-n] = one-hot vector of attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if feat in originAttr:
            X.append(1)
        else:
            X.append(0)
    attribute_count = len(self.lin_attribute_headers)

    # Binned length
    self.FEATURES.append('keywords -1-3')
    if -1 < len(word) <= 3: # 0.10805918674946825
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 3-4')
    if 3 < len(word) <= 4: # 0.2557921363700965
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 4-5')
    if 4 < len(word) <= 5: # 0.22150869117816596
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 5-6')
    if 5 < len(word) <= 6: # 0.19106959103244378
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 6-8')
    if 6 < len(word) <= 8: # 0.16618336595663957
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('keywords 8-10000000')
    if 8 < len(word) <= 10000000: # 0.057387028713185934
    	X.append(1)
    else:
    	X.append(0)

    length_count = len(X) - attribute_count - idf_bucket_count

    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}

    if tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequncy() for signal in self.signalIndex[tupleID]])

    self.FEATURES.append('maximum_TF_normalized')
    normTF = (signal.getTermFrequncy() / self.maximumTF[tupleID])
    X.append(normTF)

    # Term Frequency normalized single attribute
    self.FEATURES.append('0 < normTF <= 0.15')
    if 0 < normTF <= 0.15: # 0.045329723115660955
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.15 < normTF <= 0.2')
    if 0.15 < normTF <= 0.2: # 0.08391560191067456
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.2 < normTF <= 0.25')
    if 0.2 < normTF <= 0.25: # 0.10563302531808486
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.25 < normTF <= 0.34')
    if 0.25 < normTF <= 0.34: # 0.19701010295708057
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.34 < normTF <= 0.5')
    if 0.34 < normTF <= 0.5: # 0.31634064709430953
    	X.append(1)
    else:
    	X.append(0)
    self.FEATURES.append('0.5 < normTF <= 1')
    if 0.5 < normTF <= 1: # 0.2517708996041895
    	X.append(1)
    else:
    	X.append(0)

    for feat in self.lin_attribute_headers:  # Term frequency among attributes
        self.FEATURES.append('TF_{}_normalized'.format(feat))
        if feat in originAttr:
            X.append(originCounts[originAttr.index(feat)] / sum(originCounts))
        else:
            X.append(0)

    # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+') # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)
    non_alpha_count = 1

    # Original token features
    isTitle = 0
    isUpper = 0
    noun = 0
    verb = 0
    adjective = 0
    adverb = 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')  #
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')  #
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)
    
    return X
