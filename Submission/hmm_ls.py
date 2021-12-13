import sys
import numpy as np
import math
from EvalScript import evalResult


class HMM_LS():

    unk_token = "#UNK#"
    alpha = math.pow(10, -10)

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

        self.tokens = []
        self.labels = []
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
        for i in range(len(e_table)):
            e_table[i] = (e_table[i] + self.alpha) / \
                (np.sum(e_table, axis=1)[i]+self.alpha*len(e_table))

        self.e_table = e_table
        return e_table

    def estimate_q(self):
        q_table = np.zeros(
            (len(self.unique_labels)+1, len(self.unique_labels)+1))

        rows = ['START'] + self.unique_labels
        cols = self.unique_labels + ['STOP']

        for labels in self.labels:
            labels.insert(0, 'START')
            labels.append('STOP')

            for i in range(len(labels)-1):
                cur_label = labels[i]
                next_label = labels[i+1]
                q_table[rows.index(cur_label)][cols.index(next_label)] += 1

        for i in range(len(q_table)):
            q_table[i] = (q_table[i] + self.alpha) / \
                (np.sum(q_table, axis=1)[i]+self.alpha*(len(q_table)-1))

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

            for uIndex in range(0, m):
                current_e = self.e_table[uIndex,
                                         self.unique_tokens.index(cur_word)]
                if (j == 0):
                    current_q = self.q_table[0, uIndex]
                    pi[j+1, uIndex] = 1 * current_e * current_q
                else:
                    max_prob = 0
                    for vIndex in range(0, m):
                        current_q = self.q_table[vIndex+1, uIndex]
                        cur_prob = pi[j, vIndex] * current_e * current_q

                        if (cur_prob > max_prob):
                            max_prob = cur_prob
                    pi[j+1, uIndex] = max_prob

        # Termination step
        max_prob = 0
        for vIndex in range(0, m):  # v = state1,state2,...statem
            current_q = self.q_table[vIndex+1, -1]
            cur_prob = pi[n, vIndex] * current_q
            if (cur_prob > max_prob):
                max_prob = cur_prob
        pi[n+1, -1] = max_prob

        # Backward algorithm
        yStar = [self.unique_labels.index("O")]*(n+1)
        max_prob = 0

        for uIndex in range(0, m):
            current_q = self.q_table[uIndex+1, -1]
            cur_prob = pi[n, uIndex] * current_q

            if (cur_prob > max_prob):
                max_prob = cur_prob
                yStar[n] = uIndex

        for j in range(n-1, 0, -1):
            max_prob = 0
            for uIndex in range(0, m):
                current_q = self.q_table[uIndex+1, yStar[j+1]]
                cur_prob = pi[j, uIndex] * current_q
                if (cur_prob > max_prob):
                    max_prob = cur_prob
                    yStar[j] = uIndex

        labelled_preds = [self.unique_labels[y] for y in yStar[1:]]
        return labelled_preds

    def predict_p4(self, input_path, output_path):
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


# Part 4

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please make sure you have installed Python 3.4 or above!\nAvailable datasets: ES or RU')
        print("Usage on Windows:  python hmm.py <dataset>")
        print("Usage on Linux/Mac:  python3 hmm.py <dataset>")
        sys.exit()

    directory = f'{sys.argv[1]}/{sys.argv[1]}'

    hmm = HMM_LS()
    hmm.get_data(f'{directory}/train')
    hmm.estimate_e()
    hmm.estimate_q()

    hmm.predict_p4(f'{directory}/dev.in', f'{directory}/dev.p4.out')
    hmm.predict_p4(f'{directory}/test.in', f'{directory}/test.p4.out')

    print(hmm.evaluate(directory, 'dev.p4.out'))
