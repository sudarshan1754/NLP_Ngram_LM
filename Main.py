###********************************************************************************###
  # __author__ = 'sid'                                                             #
  # This program is written as part of the Natural Language Processing Home Work 1 #
  # @copyright: Sudarshan Sudarshan (Sid)                                          #
###********************************************************************************###

import nltk

class NLPAssignment:

    def tokenization(self, fpath):

        # list for unigrams and bigrams and the total number of tokens (N)
        uni_bigrams = []

        # Store the < word: frequency> in dictonary
        word_frequency_unigrams = {}
        word_frequency_bigrams = {}

        # A variable to stores the totalwordcount
        No_of_Words = 0


        # open the file and read the contents of the file line by line and then create a list of all tokens
        # Once the list has been created, for each token in the list count the no. of occurences of that token
        # and store this in the dictionary data structure

        file_content = open(fpath)
        for line in file_content.readlines():
            # tokens = []
            tokens = nltk.WhitespaceTokenizer().tokenize(line)

            for index, token in enumerate(tokens):  # Create the dictionary

                # Increment the No_of_Words by 1
                No_of_Words += 1

                # for unigrams
                if token in word_frequency_unigrams:
                    word_frequency_unigrams[token] += 1
                else:
                    word_frequency_unigrams[token] = 1

                # for bigrams
                if index < len(tokens) - 1:
                    if (tokens[index] + " " + tokens[index + 1]) in word_frequency_bigrams:
                        word_frequency_bigrams[(tokens[index] + " " + tokens[index + 1])] += 1
                    else:
                        word_frequency_bigrams[(tokens[index] + " " + tokens[index + 1])] = 1

        # Store the Unigrams dictionary, bigrams dictionary and No of tokens (N) in a list so that it can be returned
        uni_bigrams.append(word_frequency_unigrams)
        uni_bigrams.append(word_frequency_bigrams)
        uni_bigrams.append(No_of_Words)

        return uni_bigrams

    def Unigram_Probability(self, unigrams, No_of_Words):

        # A dictionary to store the <unigram: probability>
        unigram_probability = {}

        for word, count in unigrams.items():
            # Use MLE Estimation
            unigram_probability[word] = count / float(No_of_Words)

        return unigram_probability

        # test
        # sum = 0.0
        # for key, prob in unigram_probability.items():
        # # sum += prob
        #     print str(key) + " " + str(prob)
        # # print unigram_probability
        # # print sum

    def Bigram_Probability(self, unigrams, bigrams, discount_factor):

        # get the count of N1 (No. of bigrams has occuring once) and N2 (No. of bigrams has occuring twice)
        N1 = 0
        N2 = 0
        for word, count in bigrams.items():
            if count == 1:
                N1 += 1
            elif count == 2:
                N2 += 1

        # A dictionary to store the <unigram: probability>
        bigram_probability = {}

        # calculate the probability using MLE, Discount factor and Good turing
        for word, count in bigrams.items():
            # if count = 1, apply Good Turing and discounting factor
            if count == 1:
                bigram_probability[word] = ((2 * (N2 / (float(N1)))) / unigrams[word.split(" ")[0]]) * discount_factor
            # if count > 1, apply basic MLE and discount factor
            else:
                bigram_probability[word] = (count / float(unigrams[word.split(" ")[0]])) * discount_factor

        # test
        # sum = 0.0
        # for key, prob in bigram_probability.items():
        #     print key, prob
        #     sum += prob
        #     # print str(key) + " " + str(prob)
        # # print unigram_probability
        # print sum

        return bigram_probability

    def BackOffCalculation(self, unigrams, bigrams, unigram_probability, bigram_probability):

        backoffCoefficients = {}

        for unig in unigrams:
            probSumGT = 0.0
            probSumNonGT = 0.0
            probDenom = 0.0
            for unig1 in unigrams:
                newWord = unig + " " + unig1
                if newWord in bigrams:
                    if bigrams[newWord] == 1:
                        probSumGT += bigram_probability[newWord]
                    else:
                        probSumNonGT += bigram_probability[newWord]
                    probDenom += unigram_probability[unig1]

            backoffCoefficients[unig] = (1 - (probSumGT + probSumNonGT)) / (1 - probDenom)


        # for bigram, count in bigrams.items():
        #     history = bigram.split(" ")[0]
        #
        #     probSumGT = 0.0
        #     probSumNonGT = 0.0
        #     probDenom = 0.0
        #     for unig in unigrams:
        #         newWord = history + " " + unig
        #         if newWord in bigrams:
        #             if bigrams[newWord] == 1:
        #                 probSumGT += bigram_probability[newWord]
        #             else:
        #                 probSumNonGT += bigram_probability[newWord]
        #             probDenom += unigram_probability[unig]

        for key, value in backoffCoefficients.items():
            print key, value

if __name__ == "__main__":

    # Get the file path or file name
    fpath = raw_input('Enter the file path: ')
    if len(fpath) > 0:
        call = NLPAssignment()
        call.tokenization(fpath)

        # list for unigrams (index= 0) and bigrams (index= 1) and the total number (index= 2)of tokens (N)
        uni_bigrams = []

        # call the function to calculate the wordcount of unigrams and bigrams.
        # list containing unigrams (dictionary), bigrams (dictionary) and No_of_tokens (int) will be returned
        uni_bigrams = call.tokenization(fpath)

        # call the function to calculate the probability of unigrams.
        # Dictionary containing <unigram: probability> will be returned
        unigram_probability = call.Unigram_Probability(uni_bigrams[0], uni_bigrams[2])

        # call the function to calculate the probability of bigrams.
        # Dictionary containing <unigram: probability> will be returned
        # Also multiply each bigram probability the discount factor
        discount_factor = 0.99
        bigram_probability = call.Bigram_Probability(uni_bigrams[0], uni_bigrams[1], discount_factor)

        # call the function to calculate the alpha(h), that is the back-off co-efficient
        # back_off_coefficient = \
        call.BackOffCalculation(uni_bigrams[0], uni_bigrams[1], unigram_probability, bigram_probability)

    else:  # If file path is not valid
        print "Invalid file path"
