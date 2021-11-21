import sys
from hmm import HMM

if len(sys.argv) < 2:
    print('Please make sure you have installed Python 3.4 or above!\nAvailable datasets: ES or RU')
    print("Usage on Windows:  python hmm.py <dataset>")
    print("Usage on Linux/Mac:  python3 hmm.py <dataset>")
    sys.exit()

model = HMM()
# Part 1
# Preparing Dataset
directory = f'{sys.argv[1]}/{sys.argv[1]}'
data = model.get_train_data(f'{directory}/train')
model.estimate_e(data)

test_data = model.get_test_data(f'{directory}/dev.in')
predictions = model.predict_label_p2(test_data)
model.write_devout(test_data,predictions,f'{directory}/dev.p1.out')

model.eval_script(f'{directory}/dev.p1.out', f'{directory}/dev.out')
model.estimate_q(data)


test_data = model.get_test_data(f'{directory}/dev.in')
predictions = model.viterbi(test_data)
print(predictions)
print(model.q)