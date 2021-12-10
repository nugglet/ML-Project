import re
import sys
import numpy as np
import pandas as pd
from EvalScript import evalResult

# Part 1 Functions

# Estimates emission parameters


def estimate_e(data):
    k = 1
    e = {}
    transition_count = {}
    total_count = {}
    unk_token = '#UNK#'

    # iterate each sentence
    for idx in range(len(data)):
        sentence = data[idx]
        for word in sentence:
            x, y = word[0], word[1]

            transition_count[(y, x)] = transition_count.get((y, x), 0) + 1
            total_count[y] = total_count.get(y, 0)+1

    for transition, count in transition_count.items():
        e[transition] = count / (total_count[transition[0]] + k)

    for y in total_count.keys():
        e[(y, unk_token)] = k / (total_count[y]+k)
    return e, list(total_count.keys())


# Format dataset from file
def get_data(filename):
    with open(filename, encoding='utf-8') as f:
        raw = f.read()
        # array of sentences
        sentences = raw.strip().split('\n\n')

    clean = []
    for sentence in sentences:
        words = sentence.split('\n')
        inner_array = []
        for word in words:
            token, label = word.split(' ')
            inner_array.append([token, label])
        clean.append(inner_array)
    return clean


def predict_label(filename, e, labels):
    # produces a tag y* = argmax(x|y) for each x in the sequence
    input_path = f'{filename}'
    output_path = filename.split('.')
    output_path = output_path[0] + '.p1.out'

    outp = open(output_path, 'w', encoding='utf-8')
    inp = open(input_path, encoding='utf-8')

    for line in inp.readlines():
        line = line.strip('\n')
        if not line or re.search("^\s*$", line):
            outp.write("\n")
        else:
            argmax = 0

            # check if it is a known word
            ystar = None
            for label in labels:
                value = e.get((label, line), 0)
                if value > argmax:
                    argmax = value
                    ystar = label

            # if unknown word
            if ystar == None:
                for label in labels:
                    value = e.get((label, '#UNK#'), 0)
                    if value > argmax:
                        argmax = value
                        ystar = label

            result = line + " " + ystar + "\n"
            outp.write(result)
    outp.close()
    inp.close()


def eval_script(preds, gold):
    gold = open(gold, "r", encoding='UTF-8')
    prediction = open(preds, "r", encoding='UTF-8')
    observed = evalResult.get_observed(gold)
    predicted = evalResult.get_predicted(prediction)
    return evalResult.compare_observed_to_predicted(observed, predicted)

# Part 2


def estimate_q(data):
    # Estimates transition params based on MLE
    k = 1
    q = {}
    transition_count = {}
    total_count = {}

    unk_token = '#UNK#'

    # iterate each sentence
    for idx in range(len(data)):
        sentence = data[idx]

        for word_index in range(len(sentence)+1):

            if word_index < len(sentence):
                yi = sentence[word_index][1]
            else:
                yi = "STOP"

            yi_1 = sentence[word_index-1][1] if word_index-1 >= 0 else "START"
            transition_count[(yi, yi_1)] = transition_count.get(
                (yi, yi_1), 0)+1
            total_count[yi_1] = total_count.get(yi_1, 0)+1

    for transition, count in transition_count.items():
        q[transition] = count/total_count[transition[1]]

    # print(q)
    return q


def viterbi2(data, e, q, cat):
    predictions = []

    for index in range(len(data)):
        sentence = data[index]

        pi = []
        parents = []

        for word_index in range(len(sentence)+1):
            if word_index < len(sentence):
                current_word = sentence[word_index]
            else:
                current_word = "STOP"

            prev_word = sentence[word_index -
                                 1] if word_index-1 >= 0 else "START"

            node_parent = []
            node_value = []


def viterbi_5thbest(self, sentence):

    rank = 5
    # Initialisation step
    n = len(sentence)
    sentence = [None] + sentence
    m = len(self.unique_labels)
    pi = np.zeros((n+2, m, rank))

    # Forward algorithm
    for j in range(n):
        if sentence[j+1] in self.unique_tokens:
            cur_word = sentence[j+1]
        else:
            cur_word = self.unk_token

        for cur_index in range(0, m):
            current_e = self.e_table[cur_index,
                                     self.unique_tokens.index(cur_word)]
            if (j == 0):
                current_q = self.q_table[0, cur_index]
                pi[j+1, cur_index, 0] = 1 * current_e * current_q
            else:
                max_probs = []
                for prev_index in range(0, m):
                    for r in range(rank):
                        current_q = self.q_table[prev_index+1, cur_index]
                        cur_prob = pi[j, prev_index, r] * current_q
                        current_e * current_q

                        max_probs.append(cur_prob)
                max_probs.sort(reverse=True)
                if len(max_probs) > 5:
                    max_probs = max_probs[: 5]
                pi[j+1, cur_index] = max_probs

    # Termination step
    max_probs = []
    for prev_index in range(0, m):
        for r in range(rank):
            current_q = self.q_table[prev_index+1, -1]
            cur_prob = pi[-1, prev_index, r] * current_q
            max_probs.append(cur_prob)

    max_probs.sort(reverse=True)
    if len(max_probs) > 5:
        max_probs = max_probs[: 5]
    pi[n+1, -1] = max_probs

    # Backward algorithm
    yxs = []

    def take_last(elem):
        return elem[-1]

    for cur_index in range(0, m):
        for r in range(rank):

            current_q = self.q_table[cur_index+1, -1]
            cur_prob = pi[n, cur_index, r] * current_q

            yxs.append([cur_index, cur_prob])
    yxs.sort(reverse=True, key=take_last)

    def removeRepeated(lst):
        new = []
        for elem in lst:
            if elem not in new:
                new.append(elem)
        return new

    yxs = removeRepeated(yxs)

    if len(yxs) > 5:
        yxs = yxs[: 5]

    for j in range(n-1, 0, -1):
        max_probs = []
        for yx in yxs:
            for cur_index in range(0, m):
                for r in range(rank):
                    current_q = self.q_table[cur_index+1, yx[0]]
                    cur_prob = pi[j, cur_index, r] * current_q

                    max_probs.append([cur_index] + yx[: -1] + [cur_prob])
        max_probs.sort(reverse=True, key=take_last)
        yxs = max_probs
        if len(yxs) > 5:
            yxs = yxs[: 5]

    labelled_preds = [self.unique_labels[y] for y in yxs[4][:-1]]
    return labelled_preds


def viterbi(data, e, q, cat):
    predictions = []  # array for best probabilities

    error_log = []

    for idx in range(len(data)):
        sentence = data[idx]

        pi = []
        parents = []

        for word_index in range(len(sentence)+1):
            if word_index < len(sentence):
                current_word = sentence[word_index]
            else:
                current_word = "STOP"

            prev_word = sentence[word_index -
                                 1] if word_index-1 >= 0 else "START"

            # print(f'Cur {current_word}, prev {prev_word}')
            node_parent = []
            node_value = []

            for cat_index_current in range(len(cat)):
                max_pi = 0
                parent = None
                for cat_index_prev in range(len(cat)):
                    prev_pi = pi[word_index -
                                 1][cat_index_prev] if word_index-1 >= 0 else 1  # START (-1) La (0)
                    # print("HERE", cat[cat_index_current], current_word)
                    if current_word == "STOP":
                        current_e = 1
                    else:
                        try:
                            current_e = e[cat[cat_index_current], current_word]
                        except KeyError:
                            current_e = e[cat[cat_index_current], "#UNK#"]

                    try:
                        if prev_word == "START":
                            current_q = q[(cat[cat_index_current], "START")]
                        elif current_word == "STOP":
                            current_q = q[("STOP", cat[cat_index_prev])]
                        else:
                            current_q = q[(cat[cat_index_current],
                                           cat[cat_index_prev])]
                    except KeyError as error:
                        # if state sequence is not in training set
                        error_log.append(
                            (cat[cat_index_current], cat[cat_index_prev]))
                        current_q = 0
                    current_pi = prev_pi * current_q * current_e
                    if current_pi >= max_pi:
                        parent = cat_index_prev
                        max_pi = current_pi
                node_value.append(max_pi)
                node_parent.append(parent)
                # print(node_parent)
            pi.append(node_value)
            parents.append(node_parent)

        prediction = []
        for parent in reversed(parents):
            # print(parent)
            if len(prediction) == 0:
                prediction.append(parent[0])
            else:
                prediction.append(parent[prediction[-1]])
            predictions.append(prediction)

    # print(set(error_log))
    return predictions


def get_testing_data(path):
    # returns a 2D array of words and its corresponding transition state for each sentence
    with open(path, encoding='utf-8') as f:
        raw = f.read()
        # array of sentences
        sentences = raw.strip().split('\n\n')

    clean = []
    for sentence in sentences:
        # append array of words and corresponding transition
        clean.append(sentence.split('\n'))
    return clean

# Part 3


# ============= START OF SCRIPT ===============
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please make sure you have installed Python 3.4 or above!\nAvailable datasets: ES or RU')
        print("Usage on Windows:  python hmm.py <dataset>")
        print("Usage on Linux/Mac:  python3 hmm.py <dataset>")
        sys.exit()

    # Part 1
    # Preparing Dataset
    directory = f'{sys.argv[1]}/{sys.argv[1]}'
    data = get_data(f'{directory}/train')
    e, cat = estimate_e(data)
    print("cat", cat)

    # predict_label(f'{directory}/dev.in', e, cat)
    # print('TEST', eval_script(
    #    f'{directory}/dev.p1.out', f'{directory}/dev.out'))

    # Part 2
    q = estimate_q(data)
    print(q)

    # dev = open(f'{directory}/dev.in', encoding='utf-8')
    # dev_in = [line for line in dev.readlines()]
    # print(dev_in)

    test_data = get_testing_data(f'{directory}/dev.in')
    predictions = viterbi(test_data, e, q, cat)
    # print(predictions)
