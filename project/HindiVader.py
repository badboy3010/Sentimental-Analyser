from appdirs import unicode
from django.http import JsonResponse


def hindiSenti(data):
    import os
    import re
    import math
    import string
    import codecs
    import json
    from itertools import product
    from inspect import getsourcefile
    from io import open
    import pandas as pd
    # ##Constants##
    from nltk.tag import tnt
    from nltk.corpus import indian
    train_data = indian.tagged_sents('hindi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)

    # (empirically derived mean sentiment intensity rating increase for booster words)
    B_INCR = 0.293
    B_DECR = -0.293

    # (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
    C_INCR = 0.733
    N_SCALAR = -0.74

    # booster/dampener 'intensifiers' or 'degree adverbs'
    # http://en.wiktionary.org/wiki/Category:English_degree_adverbs

    BOOSTER_DICT = \
        {"बिलकुल": B_INCR, "awfully": B_INCR,
         "पूरी तरह से": B_INCR, "बड़े पैमाने पर": B_INCR, "काफी": B_INCR,
         "निश्चित रूप से": B_INCR, "गहराई से ": B_INCR, "गहरी": B_INCR, "विशाल": B_INCR,
         "विशेष रूप से": B_INCR, "असाधारण": B_INCR, "असाधारण रूप से": B_INCR,
         "चरम": B_INCR, "अत्यंत": B_INCR, "बेहद": B_INCR,
         "flipping": B_INCR, "flippin": B_INCR, "frackin": B_INCR, "fracking": B_INCR,
         "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR,
         "fuckin": B_INCR, "कमबख्त": B_INCR, "fuggin": B_INCR, "fugging": B_INCR,
         "बहुत": B_INCR, "hella": B_INCR, "अत्यधिक": B_INCR, "बेहद": B_INCR,
         "अविश्वसनीय": B_INCR, "अविश्वसनीय रूप से": B_INCR, "तीव्रता से": B_INCR,
         "प्रमुख": B_INCR, "प्रमुख रूप से": B_INCR, "मुख्य रूप से": B_INCR, "विशेष रूप से": B_INCR, "अधिक": B_INCR,
         "ज्यादा": B_INCR,
         "सब से अधिक": B_INCR, "सबसे अधिक": B_INCR, "अधिकांश": B_INCR, "अत्यन्त": B_INCR, "अति": B_INCR,
         "अधिकतम": B_INCR,
         "खासकर": B_INCR, "विशेषकर": B_INCR,
         "शुद्ध रूप से ": B_INCR, "विशुद्ध रूप से": B_INCR, "काफी": B_INCR, "वास्तव में": B_INCR,
         "उल्लेखनीय रूप से": B_INCR,
         "किसी": B_INCR, "इतना": B_INCR, "ताकि ": B_INCR, "इसलिए": B_INCR, "काफी हद तक": B_INCR, "सच में": B_INCR,
         "अच्छी तरह से": B_INCR, "कुल": B_INCR, "जबरदस्त": B_INCR,
         "uber": B_INCR, "अविश्वसनीय रूप से": B_INCR, "असामान्य रूप से ": B_INCR, "घोर": B_INCR,
         "बहुत": B_INCR,
         "लगभग": B_DECR, "मुश्किल से": B_DECR, "शायद ही": B_DECR, "बस पर्याप्त रूप से": B_DECR, "बस पर्याप्त": B_DECR,
         "थोड़े": B_DECR,
         "कम": B_DECR, "छोटी": B_DECR, "थोड़ा": B_DECR, "सीमांत": B_DECR, "मामूली": B_DECR, "कभी-कभार": B_DECR,
         "सामयिक": B_DECR, "आंशिक रूप से  ": B_DECR, "दुर्लभ": B_DECR,
         "शायद ही ": B_DECR, "कुछ": B_DECR, "इस तरह": B_DECR, "पूरी तरह": B_DECR,
         "प्रकार": B_DECR, "इस तरह से ": B_DECR, "कुछ-कुछ": B_DECR, "और भी": B_DECR}



    # check for special case idioms and phrases containing lexicon words
    SPECIAL_CASES = {"जख्म पर नमक": -3.0, "हंसी ठिठोली": -3.0, "खिल्ली उड़ा": -2.5, "उम्मीद नहीं": 0.0,
                     "खिल्ली उड़ाना": -2.5, "पर खरा नहीं": -2.5, "खतरे से बाहर": 3, "हौसले बुलंद": 3.2,
                     "बुलंद हौसले": 3.2, "जोर शोर से": 3.0, "फल फूल": 2.0

                     }
    module_dir = os.path.dirname(__file__)
    hope = os.path.join(module_dir, 'static/vader/hope.txt')
    emojika = os.path.join(module_dir, 'static/vader/emoji_utf8_lexicon.txt')

    # yeh chaiye
    def normalize(score, alpha=15):
        """
        Normalize the score to be between -1 and 1 using an alpha that
        approximates the max expected value
        """
        norm_score = score / math.sqrt((score * score) + alpha)
        if norm_score < -1.0:
            return -1.0
        elif norm_score > 1.0:
            return 1.0
        else:
            return norm_score

    # yeh chaiye
    def scalar_inc_dec(word, valence, words_and_emoticons):
        """
        Check if the preceding words increase, decrease, or negate/nullify the
        valence
        """
        previous_word = ' '
        next_word = ' '
        scalar = 0.0

        if word in BOOSTER_DICT:

            scalar = BOOSTER_DICT[word]
            if valence < 0:
                scalar *= -1

        return scalar

    class SentiText(object):
        """
        Identify sentiment-relevant string-level properties of input text.
        """

        # yeh chaiye
        def __init__(self, text):
            if not isinstance(text, str):
                text = str(text).encode('utf-8')
            self.text = text
            self.words_and_emoticons = self._words_and_emoticons()
            # doesn't separate words from\
            # adjacent punctuation (keeps emoticons & contractions)

        # yeh chaiye yeh :) :> emojis ko ':)' ':>' aisa krke bhejegaa list me STRIPPED k andr
        @staticmethod
        def _strip_punc_if_word(token):
            """
            Removes all trailing and leading punctuation
            If the resulting string has two or fewer characters,
            then it was likely an emoticon, so return original string
            (ie ":)" stripped would be "", so just return ":)"
            """
            stripped = token.strip(string.punctuation)
            if len(stripped) <= 1:
                return token
            return stripped

        # yeh chaiye yeh :) :> emojis ko ':)' ':>' aisa krke bhejegaa list me STRIPPED k andr
        def _words_and_emoticons(self):
            """
            Removes leading and trailing puncutation
            Leaves contractions and most emoticons
                Does not preserve punc-plus-letter emoticons (e.g. :D)
            """
            wes = self.text.split()
            stripped = list(map(self._strip_punc_if_word, wes))

            return stripped

    class SentimentIntensityAnalyzer(object):
        """
        Give a sentiment intensity score to sentences.
        """

        # reading te lexicons emoji plus hindi
        def __init__(self, lexicon_file=hope, emoji_lexicon=emojika):
            _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
            lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)
            with codecs.open(lexicon_full_filepath, encoding='utf-8') as f:
                self.lexicon_full_filepath = f.read()
            self.lexicon = self.make_lex_dict()

            emoji_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), emoji_lexicon)
            with codecs.open(emoji_full_filepath, encoding='utf-8') as f:
                self.emoji_full_filepath = f.read()
            self.emojis = self.make_emoji_dict()

        # making dicitonary of lexicons
        def make_lex_dict(self):
            """
            Convert lexicon file to a dictionary
            """
            lex_dict = {}
            for line in self.lexicon_full_filepath.rstrip('\n').split('\n'):
                if not line:
                    continue
                (word, measure) = line.strip().split('\t')[0:2]
                lex_dict[word] = float(measure)

            return lex_dict

        # makinf dicitonary of emojis
        def make_emoji_dict(self):
            """
            Convert emoji lexicon file to a dictionary
            """
            emoji_dict = {}
            for line in self.emoji_full_filepath.rstrip('\n').split('\n'):
                (emoji, description) = line.strip().split('\t')[0:2]
                emoji_dict[emoji] = description

            return emoji_dict

        def polarity_scores(self, text):
            """
            Return a float for sentiment strength based on the input text.
            Positive values are positive valence, negative value are negative
            valence.
            """
            # convert emojis to their textual descriptions
            text_no_emoji = ""
            prev_space = True
            for chr in text:
                if chr in self.emojis:
                    # get the textual description
                    description = self.emojis[chr]
                    if not prev_space:
                        text_no_emoji += ' '
                    text_no_emoji += description
                    prev_space = False
                else:
                    text_no_emoji += chr
                    prev_space = chr == ' '
            text = text_no_emoji.strip()
            # ('polarity',text)
            sentitext = SentiText(text)

            sentiments = []
            tx = []
            words_and_emoticons = sentitext.words_and_emoticons
            tagged_words = (tnt_pos_tagger.tag(words_and_emoticons))
            text = [s[0] for s in tagged_words if
                    s[1].startswith('NN') or s[1].startswith('PREP') or s[1].startswith('NNP') \
                    or s[1].startswith('NNC')]
            tx.extend(text)

            for i, item in enumerate(words_and_emoticons):
                valence = 0

                # check for vader_lexicon words that may be used as modifiers or negations
                if item.lower() in BOOSTER_DICT:
                    sentiments.append(valence)

                    continue
                if item.lower() == "नहीं" and words_and_emoticons[i - 1] not in ['से', 'को'] and words_and_emoticons[
                    i - 1] in tx:
                    sentiments.append(valence)

                    continue

                if (i < len(words_and_emoticons) - 2):
                    if (item.lower() == "सच" \
                            and words_and_emoticons[i + 1].lower() == "में"):
                        sentiments.append(valence)

                        continue

                sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)
                (sentiments)
            sentiments = self._but_check(words_and_emoticons, sentiments)
            valence_dict = self.score_valence(sentiments, text)

            return valence_dict

        def sentiment_valence(self, valence, sentitext, item, i, sentiments):

            words_and_emoticons = sentitext.words_and_emoticons
            item_lowercase = item.lower()
            if item_lowercase in self.lexicon:
                # get the sentiment valence
                if (item_lowercase == "समस्या") and words_and_emoticons[i + 1] == "निवारण":
                    valence = self.lexicon["समस्या निवारण"]
                else:
                    valence = self.lexicon[item_lowercase]

                # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
                if item_lowercase == "नहीं" and words_and_emoticons[i - 1].lower() in self.lexicon:
                    # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                    valence = 0.0

                if (i < (len(words_and_emoticons)) - 1 and item_lowercase == "नहीं" and words_and_emoticons[
                    i + 1].lower() in self.lexicon):
                    # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                    valence = 0.0

                if item_lowercase == "नहीं" and words_and_emoticons[i - 2].lower() in self.lexicon:
                    # yaha bich me yeh tha lein wo bilkul bura nahi me error aarha tha so nikala and i != len(words_and_emoticons)-1
                    # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                    valence = 0.0

                # ['इसमें', 'कोई', 'शक', 'नहीं', 'कि', 'वह', 'प्रतिभाशाली', 'है']
                if (i < len(words_and_emoticons) - 1 and words_and_emoticons[i + 1].lower() in ['नहीं'] or \
                    words_and_emoticons[i - 1].lower() in ['नहीं', 'न']) \
                        or (i < len(words_and_emoticons) - 2 and words_and_emoticons[i + 1] != "कभी" \
                            and words_and_emoticons[i + 2].lower() == "नहीं") \
                        or (i > 2 and words_and_emoticons[i - 2].lower() == "नहीं") \
                        or (i > 7 and words_and_emoticons[i - 7].lower() in ["नहीं", "न"] and words_and_emoticons[
                    i - 2].lower() in ["या", "न"]):
                    valence = self.lexicon[item_lowercase] * N_SCALAR

                for start_i in range(len(words_and_emoticons) - 1, len(words_and_emoticons) - 4, -1):

                    if i > 0 and i <= len(words_and_emoticons) - 1 and words_and_emoticons[
                        i - (len(words_and_emoticons) - start_i)].lower() not in self.lexicon:

                        s = scalar_inc_dec(words_and_emoticons[i - (len(words_and_emoticons) - start_i)], valence,
                                           words_and_emoticons)
                        if start_i == len(words_and_emoticons) - 2 and s != 0:
                            s = s * 0.95

                        if start_i == len(words_and_emoticons) - 3 and s != 0:
                            s = s * 0.9

                        valence = valence + s

                        valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                        if start_i == len(words_and_emoticons) - 2:
                            valence = self._special_idioms_check(valence, words_and_emoticons, i)

                valence = self._least_check(valence, words_and_emoticons, i)
            sentiments.append(valence)
            return sentiments

        def _least_check(self, valence, words_and_emoticons, i):
            # check for negation case using "least"
            if i > 1 and words_and_emoticons[i - 1].lower() == "कम":
                valence = valence * N_SCALAR
            return valence

        @staticmethod
        def _but_check(words_and_emoticons, sentiments):
            # check for modification in sentiment due to contrastive conjunction 'but'
            words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
            if 'लेकिन' or 'मगर' or 'किन्तु' in words_and_emoticons_lower:
                if 'लेकिन' in words_and_emoticons_lower:
                    bi = words_and_emoticons_lower.index('लेकिन')
                    for sentiment in sentiments:
                        si = sentiments.index(sentiment)
                        if si < bi:
                            sentiments.pop(si)
                            sentiments.insert(si, sentiment * 0.5)
                        elif si > bi:
                            sentiments.pop(si)
                            sentiments.insert(si, sentiment * 1.5)
                elif 'मगर' in words_and_emoticons_lower:
                    bi = words_and_emoticons_lower.index('मगर')
                    for sentiment in sentiments:
                        si = sentiments.index(sentiment)
                        if si < bi:
                            sentiments.pop(si)
                            sentiments.insert(si, sentiment * 0.5)
                        elif si > bi:
                            sentiments.pop(si)
                            sentiments.insert(si, sentiment * 1.5)
                elif 'किन्तु' in words_and_emoticons_lower:
                    bi = words_and_emoticons_lower.index('किन्तु')
                    for sentiment in sentiments:
                        si = sentiments.index(sentiment)
                        if si < bi:
                            sentiments.pop(si)
                            sentiments.insert(si, sentiment * 0.5)
                        elif si > bi:
                            sentiments.pop(si)
                            sentiments.insert(si, sentiment * 1.5)

            return sentiments

        @staticmethod
        def _special_idioms_check(valence, words_and_emoticons, i):
            words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
            onezero = "{0} {1}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i - 1])

            twoonezero = "{0} {1} {2}".format(words_and_emoticons_lower[i - 2],
                                              words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

            twoone = "{0} {1}".format(words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

            threetwoone = "{0} {1} {2}".format(words_and_emoticons_lower[i - 3],
                                               words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

            threetwo = "{0} {1}".format(words_and_emoticons_lower[i - 3], words_and_emoticons_lower[i - 2])

            sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]
            ("ddddddddd", sequences)
            ("outer abhi wala ", valence)
            for seq in sequences:
                if seq in SPECIAL_CASES:
                    valence = SPECIAL_CASES[seq]
                    break

            if len(words_and_emoticons_lower) - 1 > i:
                zeroone = "{0} {1}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1])
                if zeroone in SPECIAL_CASES:
                    valence = SPECIAL_CASES[zeroone]
                    ("special casse", zeroone, SPECIAL_CASES[zeroone])
            if len(words_and_emoticons_lower) - 1 > i + 1:
                zeroonetwo = "{0} {1} {2}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1],
                                                  words_and_emoticons_lower[i + 2])
                if zeroonetwo in SPECIAL_CASES:
                    valence = SPECIAL_CASES[zeroonetwo]

            # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
            n_grams = [threetwoone, threetwo, twoone]
            (n_grams)
            for n_gram in n_grams:
                if n_gram in BOOSTER_DICT:
                    (valence)
                    (n_gram)
                    valence = valence + BOOSTER_DICT[n_gram]
                    ("ngram", valence)
            ("yeh hua shyd", valence)
            return valence

        @staticmethod
        def _sentiment_laden_idioms_check(valence, senti_text_lower):
            # Future Work
            # check for sentiment laden idioms that don't contain a lexicon word
            idioms_valences = []
            for idiom in SENTIMENT_LADEN_IDIOMS:
                if idiom in senti_text_lower:
                    (idiom, senti_text_lower)
                    valence = SENTIMENT_LADEN_IDIOMS[idiom]
                    idioms_valences.append(valence)
            if len(idioms_valences) > 0:
                valence = sum(idioms_valences) / float(len(idioms_valences))
            return valence

        @staticmethod
        def _negation_check(valence, words_and_emoticons, start_i, i):
            words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
            ("negation", start_i)

            if start_i == len(words_and_emoticons) - 2:
                ("first check")
                if i < len(words_and_emoticons) - 1 and words_and_emoticons_lower[i + 1] == "कभी" and \
                        (words_and_emoticons_lower[i + 2] == "नहीं"):
                    valence = valence * 1.25
                    # ['वह', 'इतना', 'हताश', 'कभी', 'नहीं', 'था']
                elif words_and_emoticons_lower[i - 2] == "बिना" or \
                        words_and_emoticons_lower[i - 1] == "किसी":
                    valence = valence * N_SCALAR

                elif i < len(words_and_emoticons) - 1 and words_and_emoticons_lower[i + 1] == "कम":
                    valence = valence * N_SCALAR

            if start_i == len(words_and_emoticons) - 3:
                if i < len(words_and_emoticons) - 3 and words_and_emoticons_lower[i + 2] == "कभी" and \
                        (words_and_emoticons_lower[i + 3] == "नहीं"):
                    valence = valence * 1.25
                elif i > 2 and words_and_emoticons_lower[i - 3] == "बिना" and words_and_emoticons_lower[
                    i - 2] == "किसी":
                    # or words_and_emoticons_lower[i - 2] == "इसमें" and words_and_emoticons_lower[i - 1] == "कोई" :
                    valence = valence * N_SCALAR


                elif i < len(words_and_emoticons) - 2 and words_and_emoticons_lower[i + 2] == "कम":
                    valence = valence * N_SCALAR


                elif i < len(words_and_emoticons) - 3 and \
                        words_and_emoticons_lower[i + 1] == "किए" and words_and_emoticons_lower[i + 2] == "बिना":
                    valence = valence * N_SCALAR

            return valence

        def _punctuation_emphasis(self, text):
            # add emphasis from exclamation points and question marks
            ep_amplifier = self._amplify_ep(text)
            qm_amplifier = self._amplify_qm(text)
            punct_emph_amplifier = ep_amplifier + qm_amplifier
            return punct_emph_amplifier

        @staticmethod
        def _amplify_ep(text):
            # check for added emphasis resulting from exclamation points (up to 4 of them)
            ep_count = text.count("!")
            if ep_count > 4:
                ep_count = 4
            # (empirically derived mean sentiment intensity rating increase for
            # exclamation points)
            ep_amplifier = ep_count * 0.292
            return ep_amplifier

        @staticmethod
        def _amplify_qm(text):
            # check for added emphasis resulting from question marks (2 or 3+)
            qm_count = text.count("?")
            qm_amplifier = 0
            if qm_count > 1:
                if qm_count <= 3:
                    # (empirically derived mean sentiment intensity rating increase for
                    # question marks)
                    qm_amplifier = qm_count * 0.18
                else:
                    qm_amplifier = 0.96
            return qm_amplifier

        @staticmethod
        def _sift_sentiment_scores(sentiments):
            # want separate positive versus negative sentiment scores
            pos_sum = 0.0
            neg_sum = 0.0
            neu_count = 0
            for sentiment_score in sentiments:
                if sentiment_score > 0:
                    pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
                if sentiment_score < 0:
                    neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
                if sentiment_score == 0:
                    neu_count += 1
            return pos_sum, neg_sum, neu_count

        def score_valence(self, sentiments, text):
            if sentiments:
                sum_s = float(sum(sentiments))
                # (sum_s)
                # compute and add emphasis from punctuation in text
                punct_emph_amplifier = self._punctuation_emphasis(text)
                if sum_s > 0:
                    sum_s += punct_emph_amplifier
                elif sum_s < 0:
                    sum_s -= punct_emph_amplifier

                compound = normalize(sum_s)
                # (compound)
                # discriminate between positive, negative and neutral sentiment scores
                pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

                if pos_sum > math.fabs(neg_sum):
                    pos_sum += punct_emph_amplifier
                elif pos_sum < math.fabs(neg_sum):
                    neg_sum -= punct_emph_amplifier

                total = pos_sum + math.fabs(neg_sum) + neu_count
                pos = math.fabs(pos_sum / total)
                neg = math.fabs(neg_sum / total)
                neu = math.fabs(neu_count / total)

            else:
                compound = 0.0
                pos = 0.0
                neg = 0.0
                neu = 0.0

            sentiment_dict = \
                {"neg": round(neg, 3),
                 "neu": round(neu, 3),
                 "pos": round(pos, 3),
                 "compound": round(compound, 4)}

            return sentiment_dict

    analyzer = SentimentIntensityAnalyzer()

    ss = analyzer.polarity_scores(data)
    print("{:-<65} {}".format(data, str(ss)))
    if (ss['compound'] >= 0.05):
        polarity = "Positive"
    elif (ss['compound'] <= -0.05):
        polarity = "Negative"
    else:
        polarity = "Neutral"

    arraylist = ['Review: ' + data, 'Negative: ' + str(ss['neg']), 'Neutral: ' + str(ss['neu']),
                 'Positive: ' + str(ss['pos']), \
                 'Compound: ' + str(ss['compound']), 'Overall Sentiment: ' + polarity]
    print(arraylist)
    # --- examples -------

    return JsonResponse(arraylist, safe=False)