from nltk.translate.bleu_score import corpus_bleu
from jiwer import wer


def generate_outputs(func) -> list:
    """
    generate outputs for the test set
    :param func: a python function that converts gendered sentences to be gender-neutral
    :return: a list of gender-neutral sentences
    """
    with open('test_set/source.txt', 'r') as f:
        source = f.readlines()

    return [func(sentence) for sentence in source]


def get_bleu(generation: list, annotation: list) -> float:
    """
    find the BLEU score given a list of generations and annotations
    :param generation: generations as a list of sentences (str)
    :param annotation: annotations as a list of sentences (str)
    :return: BLEU score from 0 - 100
    """
    annotation_lists = [[ann] for ann in annotation]  # corpus_bleu takes in a list of lists for the reference
    bleu = corpus_bleu(annotation_lists, generation)
    return bleu * 100


def get_word_error_rate(generation: list, annotation: list) -> float:
    """
    find the word error rate (word level Levenshtein distance) given a list of generations and annotations
    :param generation: generation output as a list of sentences (str)
    :param annotation: annotation as a list of sentences (str)
    :return: word error rate (can be larger than 100, although this happens infrequently)
    """
    error = wer(annotation, generation)
    return error * 100


def read_file(file: str) -> list:
    """
    read the contents of a file
    :param file: the path and filename
    :return: a list of lines (str) from the file
    """
    with open(file, 'r') as f:
        return f.readlines()


def get_metrics(generation_file: str, annotation_file: str) -> dict:
    """
    get the bleu score and word error rate for a corresponding generation and annotation
    :param generation_file: the path and filename containing generations
    :param annotation_file: the path and filename containing annotations
    :return: a dictionary with the bleu score and word error rate
    """
    generation = read_file(generation_file)
    annotation = read_file(annotation_file)

    bleu = get_bleu(generation, annotation)
    word_error_rate = get_word_error_rate(generation, annotation)

    return {
        'bleu': bleu,
        'word_error_rate': word_error_rate
    }