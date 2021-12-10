import numpy as np


class HMM():
    def get_data(self, path, is_labelled=True):
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
                token, label = pair.split(' ')
                inner_tokens.append(token)
                inner_labels.append(label)

            self.tokens.append(inner_tokens)
            self.labels.append(inner_labels)

        self.unique_tokens = self.get_unique(self.tokens)
        self.unique_labels = self.get_unique(self.labels)

    def get_unique(self, nested_list):
        flattened_list = [item for sublist in nested_list for item in sublist]
        return sorted(list(set(flattened_list)))

    # Part 1
    # Estimate the emission table
    def estimate_e(self):
        e_table = np.zeros((len(self.unique_labels), len(self.unique_tokens)))
        for _, (token, label) in enumerate(zip(self.tokens, self.labels)):
            for _, (word, pos) in enumerate(zip(token, label)):
                e_table[self.unique_labels.index(
                    pos)][self.unique_tokens.index(word)] += 1

        e_table /= e_table.sum(axis=1)[:, np.newaxis]
        return e_table

    # Part 2
    # Outputs a 8x8 matrix of the transition table in the format
    # Matrix is indexed by position number and not label i.e. [0,0] -> START-B-neg
    # []        [B-negative][B-neutral][B-positive][I-negative][I-neutral][I-positive][O][STOP]
    # [START]
    # [B-negative]
    # [B-neutral]
    # [B-positive]
    # [I-negative]
    # [I-neutral]
    # [I-positive]
    # [O]

    def estimate_q(self):

        cats = self.unique_labels.copy()
        cats.insert(0, "START")
        cats.append("STOP")
        self.cats = cats

        q_table = np.zeros(
            (len(cats), len(cats)))
        trans_counts = np.zeros(
            (len(cats), len(cats)))
        total_counts = np.zeros(len(cats))

        for sentence in self.labels:
            for i in range(len(sentence)+1):
                # get index positions of labels
                yi = cats.index(sentence[i]) if i < len(
                    sentence) else cats.index("STOP")
                yi_1 = cats.index(sentence[i-1]) if i - \
                    1 >= 0 else cats.index("START")

                trans_counts[yi_1, yi] += 1
                total_counts[yi_1] += 1

        for yi_1 in range(0, len(cats)-1):
            for yi in range(1, len(cats)):
                q_table[yi_1, yi] = trans_counts[yi_1, yi]/total_counts[yi_1]
        # remove START column and STOP row
        q_table = q_table[:-1, 1:]
        return q_table

    # Part 1
    # Predict labels using the emission table
    def predict_e(self, e_table, input_path, output_path):
        predict_label = []
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

    # Part 2
    # Predict labels using both emission and transition tables with the viterbi algorithm
    def viterbi(self, q, e, Pi=None):
        # https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm

        K = q.shape[0]
        Pi = Pi if Pi is not None else np.full(K, 1/K)
        T = len(self.unique_labels)
        # the probability of the most likely path so far
        T1 = np.empty((K, T))
        T2 = np.empty((K, T))  # the x_j-1 of the most likely path so far

        # Initilaize the tracking tables from first observation
        T1[:, 0] = Pi * e[:, self.unique_labels[0]]
        2[:, 0] = 0

        # Iterate through the observations updating the tracking tables
        for i in range(1, T):
            T1[:, i] = np.max(T1[:, i - 1] * q.T *
                              e[np.newaxis, :, self.unique_labels[i]].T, 1)  # cats or unique l
            T2[:, i] = np.argmax(T1[:, i - 1] * q.T, 1)

        # Build the output, optimal model trajectory
        x = np.empty(T)
        x[-1] = np.argmax(T1[:, T - 1])
        for i in reversed(range(1, T)):
            x[i - 1] = T2[x[i], i]

        return x, T1, T2


hmm = HMM()
es_directory = 'ES/ES'
ru_directory = 'RU/RU'
hmm.get_data(f'{es_directory}/train')
e = hmm.estimate_e()
q = hmm.estimate_q
# print(hmm.unique_tokens)
# print(hmm.estimate_q())
# print(hmm.viterbi())
x, T1, T2 = hmm.viterbi(q=q, e=e)
print(x)
