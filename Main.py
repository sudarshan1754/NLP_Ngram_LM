# ##********************************************************************************###
# __author__ = 'sid'                                                             #
# This program is written as part of the Natural Language Processing Home Work 1 #
# @copyright: Sudarshan Sudarshan (Sid)                                          #
# ##********************************************************************************###

import nltk
import time
import math


class NLPAssignmentTraining:
    # Function to tokenize the file and create the <unigram: count>, <bigram: count> and total no of tokens

    def tokenization(self, fpath):

        # list for unigrams and bigrams and the total number of tokens (N)
        uni_bigrams = []

        # Store the < word: frequency> in dictionary
        word_frequency_unigrams = {}
        word_frequency_bigrams = {}

        # A variable to store the total word count
        No_of_Words = 0


        # open the file and read the contents of the file line by line and then create a list of all tokens
        # Once the list has been created, for each token in the list count the no. of occurrences of that token
        # and store this in the dictionary data structure

        file_content = open(fpath)
        for line in file_content.readlines():

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

    # Function to calculate the Unigram probability

    def Unigram_Probability(self, unigrams, No_of_Words):

        # A dictionary to store the <unigram: probability>
        unigram_probability = {}

        for word, count in unigrams.items():
            # Use MLE Estimation
            unigram_probability[word] = math.log((count / float(No_of_Words)), 2)  # storing the log probabilities

        return unigram_probability

    # Function to calculate the Bigram Probability

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
                bigram_probability[word] = math.log(
                    (((2 * (N2 / (float(N1)))) / unigrams[word.split(" ")[0]]) * discount_factor),
                    2)  # storing the log probabilities
            # if count > 1, apply basic MLE and discount factor
            else:
                bigram_probability[word] = math.log(((count / float(unigrams[word.split(" ")[0]])) * discount_factor),
                                                    2)  # storing the log probabilities

        return bigram_probability

    # Function to calculate Backoff Weights

    def BackOffCalculation(self, unigrams, bigrams, unigram_probability, bigram_probability):

        # A dictionary to store <word: backoff_wgt>
        backoffWgts = {}

        # for each bigram (unigram(h) + unigram(w)) check if there is a bigram
        for unig in unigrams:

            probSumGT = 0.0  # To store the probability of bigrams obtained from GT
            probSumNonGT = 0.0  # To store the probability of bigrams obtained from Non-GT
            probDenom = 0.0  # To store the probability of current word present in bigrams

            for unigram in unigrams:
                newWord = unig + " " + unigram
                if newWord in bigrams:
                    if bigrams[newWord] == 1:
                        probSumGT += bigram_probability[newWord]
                    else:
                        probSumNonGT += bigram_probability[newWord]
                    probDenom += unigram_probability[unigram]

            backoffWgts[unig] = (1 - (probSumGT + probSumNonGT)) / (1 - probDenom)

        return backoffWgts


class NLPAssignmentTesting():

    def GetLanguageModel(self):

        lm_content = open("Language_Model")

        # To get Unigram, probability and backoffweight
        unigram_probability = {}
        unigram_backoffWgts = {}
        bigram_probability = {}

        for lNo, line in enumerate(lm_content.readlines()):
            if line == "unigrams:\n" or line == "bigrams:\n" or line == "\n":  # ignore the headers and blank lines
                pass
            else:
                if len(line.split("\t")) == 3:  # If unigram
                    unigram_probability[line.split("\t")[1]] = float(
                        line.split("\t")[0])  # Store the <unigram: probability>
                    unigram_backoffWgts[line.split("\t")[1]] = float(
                        line.split("\t")[2].rstrip('\n'))  # Store the <unigram: backoff_wgts>
                elif len(line.split("\t")) == 2:  # If bigram
                    bigram_probability[line.split("\t")[1].rstrip('\n')] = float(
                        line.split("\t")[0])  # Store the <bigram: probability>

        # A list to return all the values read from file
        lm_model = [unigram_probability, unigram_backoffWgts, bigram_probability]

        return lm_model

    def GetTestTokens(self, test_file):

        test_bigrams = {}
        test_N = 0

        test_file = open(test_file)
        for line in test_file.readlines():
            tokens = nltk.WhitespaceTokenizer().tokenize(line)
            test_N += len(tokens)  # To get the number of tokens in test set

            for index, token in enumerate(tokens):
                # for bigrams
                if index < len(tokens) - 1:
                    if (tokens[index] + " " + tokens[index + 1]) in test_bigrams:
                        test_bigrams[(tokens[index] + " " + tokens[index + 1])] += 1
                    else:
                        test_bigrams[(tokens[index] + " " + tokens[index + 1])] = 1

        test = [test_bigrams, test_N]
        return test

    def CalculatePreplexity(self, unigram_probability, unigram_backoff, bigram_probability, test_bigrams, TestWordCount):

        logSum = 0.0
        for bigram in test_bigrams:
            if bigram in bigram_probability:
                logSum += (bigram_probability[bigram] * test_bigrams[bigram])
            else:
                # Do katz Smoothing P(w|h) = alpha(h) * P(w)
                logSum += ((unigram_backoff[bigram.split(" ")[0]] * unigram_probability[bigram.split(" ")[1]]) * test_bigrams[bigram])

        logProb = (-(logSum / TestWordCount))
        perplexity = math.pow(2, logProb)

        return perplexity

if __name__ == "__main__":

    # Get the file path or file name
    fpath = raw_input('Enter the file path: ')

    # Timer to get the execution time
    start_time = time.time()

    if len(fpath) > 0:
        call = NLPAssignmentTraining()
        call.tokenization(fpath)

        # list for unigrams (index= 0) and bigrams (index= 1) and the total number (index= 2)of tokens (N)
        uni_bigrams = []

        # call the function to calculate the wordcount of unigrams and bigrams.
        # list containing unigrams (dictionary), bigrams (dictionary) and No_of_tokens (int) will be returned
        uni_bigrams = call.tokenization(fpath)

        print "Successfully extracted unigrams and bigrams....."

        # call the function to calculate the probability of unigrams.
        # Dictionary containing <unigram: probability> will be returned
        unigram_probability = call.Unigram_Probability(uni_bigrams[0], uni_bigrams[2])

        print "Successfully calculated unigram Probability....."

        # call the function to calculate the probability of bigrams.
        # Dictionary containing <unigram: probability> will be returned
        # Also multiply each bigram probability the discount factor
        discount_factor = 0.99
        bigram_probability = call.Bigram_Probability(uni_bigrams[0], uni_bigrams[1], discount_factor)

        print "Successfully calculated bigram Probability....."

        # call the function to calculate the alpha(h), that is the back-off wghts
        backoff_wgts = call.BackOffCalculation(uni_bigrams[0], uni_bigrams[1], unigram_probability, bigram_probability)

        print "Successfully calculated backoff weights for Unigrams....."

        # To store the Unigrams and bigrams
        LM_file = open("Language_Model", "w")
        LM_file.write("unigrams:\n")
        for unigram in uni_bigrams[0]:
            LM_file.write(
                str(unigram_probability[unigram]) + "\t" + str(unigram) + "\t" + str(backoff_wgts[unigram]) + "\n")

        LM_file.write("\nbigrams:\n")
        for bigram in uni_bigrams[1]:
            LM_file.write(str(bigram_probability[bigram]) + "\t" + str(bigram) + "\n")

        LM_file.close()

        print "Successfully stored Language Model in file 'Language_Model'....."

        print "\n--- %s seconds ---" % (time.time() - start_time)

        # Testing
        test = NLPAssignmentTesting()

        # To create a 3 different dictionaries and store it in a list
        # [0] <unigram: probability>
        # [1] <unigram: backoffwgt>
        # [2] <bigram: probability>
        lm_model = test.GetLanguageModel()

        # Get the name/ path of testing file
        print "\nEnter the test file:"
        test_file = raw_input()

        if len(test_file) > 0:
            # Read the testing file anf get the <bigram: count> and N
            test_bigrams = test.GetTestTokens(test_file)

            # Calculate the perplexity
            perplexity = test.CalculatePreplexity(lm_model[0], lm_model[1], lm_model[2], test_bigrams[0], test_bigrams[1])

            print "The perplexity of the test file is: " + str(perplexity)

    else:  # If file path is not valid
        print "Invalid file path"


