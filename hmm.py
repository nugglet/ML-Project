from EvalScript import evalResult

class HMM():

    e={}
    q={}
    labels = []
    unk_token = "#UNK#"
    k=1

    def estimate_e(self,data):
        count_yx = {}
        count_y = {}

        for idx in range(len(data)):
            sentence = data[idx]
            for word_label_pairs in sentence:
                x,y = word_label_pairs[0], word_label_pairs[1]

                count_yx[(y,x)] = count_yx.get((y,x),0) + 1
                count_y[y] = count_y.get(y,0)+1
        
        for transition_yx, count_yx in count_yx.items():
            self.e[transition_yx] = count_yx / (count_y[transition_yx[0]]+self.k)

        for y in count_y.keys():
            self.e[(y,self.unk_token)] = self.k / (count_y[y]+self.k)
   
        self.labels = list(count_y.keys())

    def estimate_q(self,data):
        count_yi1_yi = {}
        count_yi1 = {}

        # iterate each sentence
        for idx in range(len(data)):
            sentence = data[idx]

            for word_index in range(len(sentence)+1):

                if word_index < len(sentence):
                    yi = sentence[word_index][1]
                else:
                    yi = "STOP"

                if word_index>0:
                    yi_1 = sentence[word_index-1][1]
                else:
                    yi_1 = "START" 
                count_yi1_yi[(yi_1, yi)] = count_yi1_yi.get((yi_1, yi), 0)+1
                count_yi1[yi_1] = count_yi1.get(yi_1, 0)+1

        for transition, count in count_yi1_yi.items():
            self.q[transition] = count/count_yi1[transition[0]]

    # Get labelled dataset from file
    def get_train_data(self,path):
        with open(path, encoding='utf-8') as f:
            raw = f.read()
            # array of sentences
            sentences = raw.strip().split('\n\n')
        clean = []
        for sentence in sentences:
            words = sentence.split('\n')
            inner_array = []
            for word in words:
                inner_array.append(word.split(' '))
            clean.append(inner_array)
        return clean

    # Get unlabelled data from file
    def get_test_data(self,path):
        with open(path, encoding='utf-8') as f:
            raw = f.read()
            # array of sentences
            sentences = raw.strip().split('\n\n')
        clean = []
        for sentence in sentences:
            clean.append(sentence.split('\n'))
        return clean

    def predict_label_p2(self,data):
        predictions = []
        
        for idx in range(len(data)):
            sentence = data[idx]
            inner_pred = []
            for word in sentence:
                ystar = None
                argmax = 0
                for y in self.labels:
                    #print(y)
                    #print(word)
                    try:
                        value = self.e.get((y,word),0)
                    except KeyError:
                        value = self.e.get((y, self.unk_token), 0)
                    if value > argmax:
                        argmax = value
                        ystar = y

                inner_pred.append(ystar)
            predictions.append(inner_pred)
        return predictions
    
    def write_devout(self,data,label,output_path):
        with open(output_path, 'w', encoding='utf-8') as outp:
            for i, (xs,ys) in enumerate(zip(data,label)):
                for i, (x,y) in enumerate(zip(xs,ys)):
                    text = f"{x} {y}\n"
                    outp.write(text)
                outp.write('\n')

    def eval_script(self,preds, gold):
        gold = open(gold, "r", encoding='UTF-8')
        prediction = open(preds, "r", encoding='UTF-8')
        observed = evalResult.get_observed(gold)
        predicted = evalResult.get_predicted(prediction)
        return evalResult.compare_observed_to_predicted(observed, predicted)

    def viterbi(self,data):
        predictions = []  # array for best probabilities

        for idx in range(len(data)):
            sentence = data[idx]

            pi = []
            parents = []

            for word_index in range(len(sentence)+1):
                if word_index < len(sentence):
                    current_word = sentence[word_index]
                else:
                    current_word = "STOP"

                if word_index>0:
                    prev_word = sentence[word_index -1]
                else:
                    prev_word = "START" 

                node_parent = []
                node_value = []
                for cur_label_index in range(len(self.labels)):
                    max_pi = 0
                    parent = None
                    for prev_label_index in range(len(self.labels)):
                        if prev_word=="START":
                            prev_pi = 1
                        else:
                            prev_pi = pi[word_index-1][prev_label_index] 
                        
                        if current_word == "STOP":
                            current_e = 1
                        else:
                            try:
                                current_e = self.e[self.labels[cur_label_index], current_word]
                            except KeyError:
                                current_e = self.e[self.labels[cur_label_index], self.unk_token]

                        try:
                            if current_word == "START":
                                current_q = self.q[( "START",self.labels[self.labels])]
                            elif current_word == "STOP":
                                current_q = self.q[(self.labels[prev_label_index],"STOP")]
                            else:
                                current_q = self.q[(self.labels[prev_label_index],
                                            self.labels[cur_label_index])]
                        except KeyError as error:
                            # print(error)
                            current_q = 0
                        current_pi = prev_pi * current_q * current_e
                        if current_pi > max_pi:
                            parent = prev_label_index
                            max_pi = current_pi
                    node_value.append(max_pi)
                    node_parent.append(parent)
                pi.append(node_value)
                parents.append(node_parent)
           
            prediction = []
            for parent in reversed(parents):
                if len(prediction) == 0:
                    prediction.append(parent[0])
                else:
                    prediction.append(parent[prediction[-1]])

            predictions.append(prediction)
        return predictions

            