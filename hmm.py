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

    return e, total_count.keys()


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


def evaluate_model(preds, gold):

    preds_data = get_data(preds)
    gold_data = get_data(gold)

    correct_preds = 0
    pred_count = 0
    gold_count = 0

    for sentence_index in range(len(preds_data)):

        pred_count += len(preds_data[sentence_index])
        gold_count += len(gold_data[sentence_index])

        for record_index in range(len(preds_data[sentence_index])):
            pred_record = preds_data[sentence_index][record_index]
            gold_record = gold_data[sentence_index][record_index]

            if pred_record == gold_record:
                correct_preds += 1

    print(correct_preds, pred_count, gold_count)
    # total no of correctly predicted entities/total no of predicted entities
    precision = correct_preds / pred_count
    # total no of correctly predicted entities/total no of gold entities
    recall = correct_preds / gold_count
    F = 2/((1/precision)+(1/recall))
    print(precision, recall)
    return F


def eval_script(preds, gold):
    gold = open(gold, "r", encoding='UTF-8')
    prediction = open(preds, "r", encoding='UTF-8')
    observed = evalResult.get_observed(gold)
    predicted = evalResult.get_predicted(prediction)
    return evalResult.compare_observed_to_predicted(observed, predicted)

# Part 2
# NOT DONE SOZ


def estimate_q(data):
    k = 1
    q = {}
    transition_count = {}
    total_count = {}
    unk_token = '#UNK#'

    # iterate each sentence
    for idx in range(len(data)):
        sentence = data[idx]
        for word in sentence:
            yi, yi_1 = word[0], word[1]

            transition_count[(y, x)] = transition_count.get((y, x), 0) + 1
            total_count[y] = total_count.get(y, 0)+1

    for transition, count in transition_count.items():
        e[transition] = count / (total_count[transition[0]] + k)

    for y in total_count.keys():
        e[(y, unk_token)] = k / (total_count[y]+k)

    return e, total_count.keys()


# ============= START OF SCRIPT ===============


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please make sure you have installed Python 3.4 or above!\nAvailable datasets: ES or RU')
        print("Usage on Windows:  python hmm.py <dataset>")
        print("Usage on Linux/Mac:  python3 hmm.py <dataset>")
        sys.exit()

    # Part 1
    # Preparing Dataset
    directory = f'{sys.argv[2]}/{sys.argv[2]}'
    data = get_data(f'{directory}/train')
    e, cat = estimate_e(data)
    predict_label(f'{directory}/dev.in', e, cat)
    print(eval_script(f'{directory}/dev.p1.out', f'{directory}/dev.out'))
