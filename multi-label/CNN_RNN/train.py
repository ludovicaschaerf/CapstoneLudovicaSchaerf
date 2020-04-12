import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import json
from torch.utils.data import DataLoader
from build_vocab import Vocabulary
from data_loader import get_loader
from model_attention import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    print("load vocabulary ...")    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print("build data loader ...")
    # Build data loader
    with open(args.caption_path, 'r') as infile:
        caption = json.load(infile)
    with open(args.idx2path_path, 'r') as infile:
        idx2path = json.load(infile)
    caption_train = {}
    caption_test = {}
    idx2path_train = {}
    idx2path_test = {}
    for i,capt in enumerate(caption):
        if i > 20000:
            caption_test[capt] = caption[capt]
            idx2path_test[str(i-20000)] = idx2path[str(i)]
            continue
        caption_train[capt] = caption[capt]
        idx2path_train[str(i)] = idx2path[str(i)]
    print(len(caption_test.keys()), len(caption_train.keys()), len(idx2path_test.keys()), len(idx2path_train.keys()))
    
    data_loader_train = get_loader(args.image_dir, idx2path_train, caption_train, vocab, 
                                 args.batch_size, shuffle=True, transform=transform,
                                 num_workers=args.num_workers) 
   
    
    data_loader_test = get_loader(args.image_dir, idx2path_test, caption_test, vocab, 
                                 args.batch_size, shuffle=True, transform=transform,
                                 num_workers=args.num_workers) 
    
    print("build the models ...")
    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    #encoder.load_state_dict(torch.load("models/encoder-2-1000.ckpt"))
    #decoder.load_state_dict(torch.load("models/decoder-2-1000.ckpt")) 


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters())# + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader_train)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths, path) in enumerate(data_loader_train):
            print(i)
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
    
    dict_test = {}
    for i, (images, captions, lengths, path) in enumerate(data_loader_test):
            print(i)
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
            dict_test[path] = (outputs, target)
    
    with open('../results/predictions_CNN_RNN.pkl', 'wb') as outfile:
        pickle.dump(dict_test, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../results/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='../../data/results/zh_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../../../data_tate', help='directory for resized images') #data/resized2014
    parser.add_argument('--caption_path', type=str, default='../../data/results/MyData.json', help='path for train annotation json file')
    parser.add_argument('--idx2path_path', type=str, default='../../data/results/idx2path.json', help='path for train idx2path json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=700, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
