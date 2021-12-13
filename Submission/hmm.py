import sys
import numpy as np
import math

from EvalScript import evalResult
import copy


class HMM():

    unk_token = "#UNK#"
    tokens = []
    labels = []
    unique_labels = []
    unique_tokens = []
    q_table = np.zeros((1))
    e_table = np.zeros((1))

    def evaluate(self, path, pred_file):
        gold = open(f'{path}/dev.out', "r", encoding='UTF-8')
        prediction = open(f'{path}/{pred_file}', "r", encoding='UTF-8')
        observed = evalResult.get_observed(gold)
        predicted = evalResult.get_predicted(prediction)
        return evalResult.compare_observed_to_predicted(observed, predicted)

    def get_data(self, path):
        with open(path, encoding='utf-8') as f:
            raw = f.read()
            # array of sentences
            sentences = raw.strip().split('\n\n')

        for sentence in sentences:
            pairs = sentence.split('\n')
            inner_tokens = []
            inner_labels = []
            for pair in pairs:
                try:
                    token, label = pair.split(' ')
                except:
                    pass
                inner_tokens.append(token)
                inner_labels.append(label)

            self.tokens.append(inner_tokens)
            self.labels.append(inner_labels)

        self.unique_tokens = self.get_unique(self.tokens)
        self.unique_tokens = self.unique_tokens + [self.unk_token]
        self.unique_labels = self.get_unique(self.labels)

    def get_test_data(self, path):
        with open(path, encoding='utf-8') as f:
            raw = f.read()
            # array of sentences
            sentences = raw.strip().split('\n\n')

        tokens = []
        for sentence in sentences:
            words = sentence.split('\n')
            tokens.append(words)

        return tokens

    def get_unique(self, nested_list):
        flattened_list = [item for sublist in nested_list for item in sublist]
        return sorted(list(set(flattened_list)))

    def estimate_e(self):
        e_table = np.zeros(
            (len(self.unique_labels), len(self.unique_tokens)+1))
        for _, (token, label) in enumerate(zip(self.tokens, self.labels)):
            for _, (word, pos) in enumerate(zip(token, label)):
                e_table[self.unique_labels.index(
                    pos)][self.unique_tokens.index(word)] += 1

        for i in range(len(self.unique_labels)):
            e_table[i, -1] += 1

        e_table /= e_table.sum(axis=1)[:, np.newaxis]
        self.e_table = e_table
        return e_table

    def predict_p1(self, input_path, output_path):
        predict_label = []
        e_table = self.e_table
        x = self.get_test_data(input_path)
        for sentence in x:
            inner_predict = []
            for word in sentence:
                if word not in self.unique_tokens:
                    word = self.unk_token
                    pred_label = e_table[:, -1]
                else:
                    pred_label = e_table[:, self.unique_tokens.index(word)]
                most_likely_label = self.unique_labels[np.argmax(pred_label)]
                inner_predict.append(most_likely_label)
            predict_label.append(inner_predict)

        with open(output_path, 'w', encoding='utf-8') as outp:
            for _, (token, label) in enumerate(zip(x, predict_label)):
                for _, (word, pos) in enumerate(zip(token, label)):
                    result = word + " " + pos + "\n"
                    outp.write(result)
                outp.write('\n')

    def estimate_q(self):
        q_table = np.zeros(
            (len(self.unique_labels)+1, len(self.unique_labels)+1))

        rows = ['START'] + self.unique_labels.copy()
        cols = self.unique_labels.copy() + ['STOP']

        for labels in self.labels:
            x = copy.deepcopy(labels)
            x.insert(0, 'START')
            x.append('STOP')

            for i in range(len(x)-1):
                cur_label = x[i]
                next_label = x[i+1]
                q_table[rows.index(cur_label)][cols.index(next_label)] += 1

        q_table /= q_table.sum(axis=1)[:, np.newaxis]
        self.q_table = q_table
        return q_table

    def viterbi(self, sentence):
        # Initialisation step
        n = len(sentence)
        sentence = [None] + sentence
        m = len(self.unique_labels)
        pi = np.zeros((n+2, m))

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
                    pi[j+1, cur_index] = 1 * current_e * current_q
                else:
                    max_prob = 0
                    for vIndex in range(0, m):
                        current_q = self.q_table[vIndex+1, cur_index]
                        cur_prob = pi[j, vIndex] * current_e * current_q

                        if (cur_prob > max_prob):
                            max_prob = cur_prob
                    pi[j+1, cur_index] = max_prob

        # Termination step
        max_prob = 0
        for prev_index in range(0, m):  # v = state1,state2,...statem
            current_q = self.q_table[prev_index+1, -1]
            cur_prob = pi[n, prev_index] * current_q
            if (cur_prob > max_prob):
                max_prob = cur_prob
        pi[n+1, -1] = max_prob

        # Backward algorithm
        y_star = [self.unique_labels.index("O")]*(n+1)
        max_prob = 0

        for cur_index in range(0, m):
            current_q = self.q_table[cur_index+1, -1]
            cur_prob = pi[n, cur_index] * current_q

            if (cur_prob > max_prob):
                max_prob = cur_prob
                y_star[n] = cur_index

        for j in range(n-1, 0, -1):
            max_prob = 0
            for cur_index in range(0, m):
                current_q = self.q_table[cur_index+1, y_star[j+1]]
                cur_prob = pi[j, cur_index] * current_q
                if (cur_prob > max_prob):
                    max_prob = cur_prob
                    y_star[j] = cur_index

        labelled_preds = [self.unique_labels[y] for y in y_star[1:]]
        return labelled_preds

    def predict_p2(self, input_path, output_path):
        total_preds = []
        count = 0

        data = self.get_test_data(input_path)

        for sentence in data:
            count += 1
            preds = self.viterbi(sentence)
            total_preds.append(preds)

        with open(output_path, 'w', encoding='utf-8') as outp:
            for _, (token, label) in enumerate(zip(data, total_preds)):
                for _, (word, pos) in enumerate(zip(token, label)):

                    result = word + " " + pos + "\n"
                    outp.write(result)
                outp.write('\n')

    # Part 3
    def viterbi_5thbest(self, sentence):

        rank = 5
        # Initialisation step
        n = len(sentence)
        sentence = [None] + sentence
        m = len(self.unique_labels)
        pi = np.zeros((n+2, m, rank))
        # pi[0,0]=1

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
                    pi[j+1, cur_index, :] = 1 * current_e * current_q
                else:
                    max_probs = []
                    for prev_index in range(0, m):
                        for r in range(rank):
                            current_q = self.q_table[prev_index+1, cur_index]
                            cur_prob = pi[j, prev_index, r] * \
                                current_e * current_q

                            max_probs.append(cur_prob)
                    max_probs.sort(reverse=True)

                    if len(max_probs) > rank:
                        max_probs = max_probs[:rank]
                    pi[j+1, cur_index] = max_probs

        # Termination step
        max_probs = []
        for prev_index in range(0, m):
            for r in range(rank):
                current_q = self.q_table[prev_index+1, -1]
                cur_prob = pi[-1, prev_index, r] * current_q
                max_probs.append(cur_prob)

        max_probs.sort(reverse=True)
        if len(max_probs) > rank:
            max_probs = max_probs[:rank]
        pi[n+1, -1] = max_probs

        # Backward algorithm
        yxs = np.zeros((n+1, rank), dtype=int) + self.unique_labels.index("O")
        max_probs = []

        def take_last(elem):
            return elem[-1]

        for prev_index in range(0, m):
            for r in range(rank):
                current_q = self.q_table[prev_index+1, -1]
                cur_prob = pi[-1, prev_index, r] * current_q

                max_probs.append([cur_index, cur_prob])
        max_probs.sort(reverse=True, key=take_last)

        def removeRepeated(lst):
            new = []
            for elem in lst:
                if elem[1] != 0 and elem not in new:
                    new.append(elem)
            return new

        max_probs = removeRepeated(max_probs)

        if len(max_probs) > rank:
            max_probs = max_probs[:rank]

        parents = [i[0] for i in max_probs]

        yxs[n, :len(max_probs)] = parents

        for j in range(n-1, 0, -1):
            max_probs = []
            for yx in yxs[j+1]:
                for cur_index in range(0, m):
                    for r in range(rank):
                        current_q = self.q_table[cur_index+1, yx]
                        cur_prob = pi[j, cur_index, r] * current_q

                        max_probs.append([cur_index, cur_prob])

            max_probs.sort(reverse=True, key=take_last)
            max_probs = removeRepeated(max_probs)

            if len(max_probs) > rank:
                max_probs = max_probs[:rank]

            parents = [i[0] for i in max_probs]
            yxs[j, :len(max_probs)] = parents

        labelled_preds = [self.unique_labels[y] for y in yxs.T[-1][1:]]

        return labelled_preds

    def predict_p3(self, input_path, output_path):
        total_preds = []
        data = self.get_test_data(input_path)

        for sentence in data:
            preds = self.viterbi_5thbest(sentence)
            total_preds.append(preds)

        with open(output_path, 'w', encoding='utf-8') as outp:
            for _, (token, label) in enumerate(zip(data, total_preds)):

                for _, (word, pos) in enumerate(zip(token, label)):

                    result = word + " " + pos + "\n"
                    outp.write(result)
                outp.write('\n')


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please make sure you have installed Python 3.4 or above!\nAvailable datasets: ES or RU')
        print("Usage on Windows:  python hmm.py <dataset>")
        print("Usage on Linux/Mac:  python3 hmm.py <dataset>")
        sys.exit()

directory = f'{sys.argv[1]}/{sys.argv[1]}'

hmm = HMM()
hmm.get_data(f'{directory}/train')
hmm.estimate_e()
hmm.estimate_q()

hmm.predict_p1(f'{directory}/dev.in', f'{directory}/dev.p1.out')
hmm.predict_p2(f'{directory}/dev.in', f'{directory}/dev.p2.out')
hmm.predict_p3(f'{directory}/dev.in', f'{directory}/dev.p3.out')

print('Evaluate on P1')
print(hmm.evaluate(directory, 'dev.p1.out'))

print('Evaluate on P2')
print(hmm.evaluate(directory, 'dev.p2.out'))

print('Evaluate on P3')
print(hmm.evaluate(directory, 'dev.p3.out'))
