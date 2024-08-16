from __future__ import division, print_function

import sys
import cv2
import codecs
import argparse

import editdistance
from dataLoader import DataLoader, Batch
from Model import Model, DecoderType
from sample_preprocessor import preprocess


class FilePaths:
    """
    Filenames and paths to data
    """
    fn_char_list = '/model/charList.txt'
    fn_accuracy = 'model/accuracy.txt'
    fn_train = 'data/'
    fn_infer = '/data/3.jpg'
    fn_corpus = '/data/hindi_vocab.txt'


def train(model, loader):
    """
    Train the neural network
    :param model: A neural network model
    :param loader: A Dataloader for the training loop
    :return:
    """
    best_error_char_rate = float("inf")
    no_improvement_since = 0
    early_stopping = 5
    epoch = 0
    while True:
        epoch += 1
        print(f"Currently on epoch {epoch}")

        # Training
        print("Training")
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            loss = model.train_batch(batch)

        # Validation
        char_error_rate = validate(model, loader)

        # If this is the least error rate, save the model parameters
        if char_error_rate < best_error_char_rate:
            print("Character error rate improved")
            no_improvement_since = 0
            model.save()
            open(FilePaths.fn_accuracy, 'w').write('Validation character error rate of saved model: %f%%' % (char_error_rate*100.0))
        else:
            print("Error rate not improved")
            no_improvement_since += 1

        # Stop training if no more improvement in the last x epochs
        if no_improvement_since > early_stopping:
            print("Training stopped")
            break


def validate(model, loader):
    model.eval()
    loader.validation_set()
    num_char_err = 0
    num_char_total = 0
    num_word_0K = 0
    num_word_total = 0

    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f"Batch: {iter_info[0], '/', iter_info[1]}")
        batch = loader.get_next()
        (recognized, _) = model.infer_batch()

        print(f"Ground Truth -> Recognized")
        for i in range(recognized):
            num_word_0K += 1 if batch.gtTexts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            num_char_err += dist
            num_char_total += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # Print the validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_0K / num_word_total
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (char_error_rate * 100.0, word_accuracy * 100.0))
    return char_error_rate


def infer(model, fn_img):
    """recognize text in image provided by file path"""
    img = preprocess(cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0].replace(" ", "") + '"')
    print('Probability:', probability[0])


def infer_by_web(path, option):
    decoder_type = DecoderType.best_path
    if option == "bestPath":
        decoder_type = DecoderType.best_path
        print("Best Path Execute")
    if option == "beamSearch":
        decoder_type = DecoderType.beam_search
    print(open(FilePaths.fn_accuracy).read())
    model = Model(codecs.open(FilePaths.fn_char_list, encoding="utf8").read(), decoder_type)
    img = preprocess(cv2.imread(path, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0].replace(" ", "") + '"')
    print('Probability:', probability[0])
    return recognized[0].replace(" ", ""), probability[0]


def main():
    """main function"""
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the NN", action="store_true")
    parser.add_argument("--validate", help="validate the NN", action="store_true")
    parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
    parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    args = parser.parse_args()

    decoder_type = DecoderType.best_path
    if args.beamsearch:
        decoder_type = DecoderType.beam_search
    elif args.wordbeamsearch:
        decoder_type = DecoderType.word_beam_search

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoader(FilePaths.fn_train, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fn_char_list, 'w', encoding='UTF-8').write(str().join(loader.char_list))

        # save words contained in dataset into file
        open(FilePaths.fn_corpus, 'w', encoding='UTF-8').write(str(' ').join(loader.train_words + loader.valid_words))

        # execute training or validation
        if args.train:
            model = Model(loader.char_list, decoder_type)
            train(model, loader)
        elif args.validate:
            model = Model(loader.char_list, decoder_type, must_restore=False)
            validate(model, loader)

    # infer text on test image
    else:
        print(open(FilePaths.fn_accuracy).read())
        model = Model(codecs.open(FilePaths.fn_char_list, encoding='utf-8').read(), decoder_type, must_restore=False)
        infer(model, FilePaths.fn_infer)


if __name__ == '__main__':
    main()

