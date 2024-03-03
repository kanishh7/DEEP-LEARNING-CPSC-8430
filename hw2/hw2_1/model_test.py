import sys
import torch
import json
from model_train import test_data, test, MODELS, encoderRNN, decoderRNN, attention
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

# Load the pre-trained model
model = torch.load('SavedModel/modelkanish.h5', map_location=lambda storage, loc: storage)

# Set the file path for the testing data and create a DataLoader
filepath = 'MLDS_hw2_1_data/testing_data/feat'
dataset = test_data('{}'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

# Load the word-to-index mapping from a pickle file
with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

# Move the model to GPU
model = model.cuda()

# Perform testing on the loaded model
ss = test(testing_loader, model, i2w)

# Write the results to the specified output file
with open(sys.argv[2], 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))

# Load the ground truth captions from the testing label file
test = json.load(open('MLDS_hw2_1_data/testing_label.json'))
output = sys.argv[2]
result = {}

# Parse the output file to create a dictionary of video IDs and generated captions
with open(output, 'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma + 1:]
        result[test_id] = caption

# Calculate BLEU scores using the method described in the paper
bleu = []
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']], captions, True))
    bleu.append(score_per_video[0])

# Calculate and print the average BLEU score
average = sum(bleu) / len(bleu)
print("Average BLEU score is " + str(average))
