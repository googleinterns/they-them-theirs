import os
import json
import re
from pathlib import Path

from evaluate_generation import generate_outputs, get_metrics

MALE_PRONOUNS = ['he', 'him', 'his', 'himself']
FEMALE_PRONOUNS = ['she', 'her', 'hers', 'herself']

SOURCE_GENERATION_ANNOTATION = [
    ['source.txt', 'generation.txt', 'target.txt'],
    ['domains/jokes.source', 'jokes.generation', 'domains/jokes.target'],
    ['domains/movie_quotes.source', 'movie_quotes.generation', 'domains/movie_quotes.target'],
    ['domains/news_articles.source', 'news_articles.generation', 'domains/news_articles.target'],
    ['domains/reddit.source', 'reddit.generation', 'domains/reddit.target'],
    ['domains/twitter.source', 'twitter.generation', 'domains/twitter.target'],
    ['genders/female.source', 'female.generation', 'genders/female.target'],
    ['genders/male.source', 'male.generation', 'genders/male.target']
]


def identity(sentence):
    return sentence


def create_outputs(func, input_folder, output_folder):
    source_file = 'source.txt'
    generation_file = 'generation.txt'

    Path(f"{input_folder}/{output_folder}").mkdir(parents=True, exist_ok=True)

    outputs = generate_outputs(source_file=os.path.join(f'{input_folder}', source_file), func=func)
    with open(f"{input_folder}/{output_folder}/{generation_file}", 'w') as f:
        for sent in outputs:
            f.write(sent)


def evaluate_outputs(eval_set, output_folder, fname):
    scores = dict()
    for file_set in SOURCE_GENERATION_ANNOTATION:
        generation_file = file_set[1]
        annotation_file = file_set[2]

        results = get_metrics(generation_file=os.path.join(eval_set, output_folder, generation_file),
                              annotation_file=os.path.join(eval_set, annotation_file))

        domain = generation_file.split('.')[0]
        scores[domain] = results

    with open(f"{eval_set}/scores/{fname}.json", 'w') as f:
        json.dump(scores, f)


def split_generation(eval_set, generation_folder, female_indices, male_indices):
    with open(os.path.join(f'{eval_set}/generations', generation_folder, 'generation.txt'), 'r') as f:
        generation = f.readlines()

    twitter = generation[:100]
    reddit = generation[100:200]
    news_articles = generation[200:300]
    movie_quotes = generation[300:400]
    jokes = generation[400:500]

    female = [generation[idx] for idx in female_indices]
    male = [generation[idx] for idx in male_indices]

    return {
        'twitter': twitter,
        'reddit': reddit,
        'news_articles': news_articles,
        'movie_quotes': movie_quotes,
        'jokes': jokes,
        'female': female,
        'male': male
    }


def is_gendered(sentence):
    sentence = sentence.lower()
    contains_male = any(re.search(r'\b{}\b'.format(m_pronoun), sentence) for m_pronoun in MALE_PRONOUNS)
    contains_female = any(re.search(r'\b{}\b'.format(f_pronoun), sentence) for f_pronoun in FEMALE_PRONOUNS)
    if contains_male and not contains_female:
        return "male"
    elif contains_female and not contains_male:
        return "female"
    return False


def main():
    # create_outputs(func=identity, input_folder='nongendered_test_set', output_folder='generations/identity')
    #
    # from jewang_neutral_converter import jewang_convert
    # create_outputs(func=jewang_convert, input_folder='nongendered_test_set', output_folder='generations/prior_work')
    #
    # from old_smart_convert import old_convert
    # create_outputs(func=old_convert, input_folder='nongendered_test_set', output_folder='generations/old_convert_1')
    #
    # from old_score_smart_convert import convert_old_score
    # create_outputs(func=convert_old_score, input_folder='nongendered_test_set', output_folder='generations/old_convert_2')
    #
    # from smart_convert import convert
    # create_outputs(func=convert, input_folder='nongendered_test_set', output_folder='generations/convert')

    # evaluate_outputs(output_folder='generations/jewang_convert', fname='jewang_convert')
    # evaluate_outputs(output_folder='generations/old_convert_1', fname='old_convert_1')
    # evaluate_outputs(output_folder='generations/old_convert_2', fname='old_convert_2')
    # evaluate_outputs(output_folder='generations/convert', fname='convert')

    # evaluate_outputs(output_folder='generations/model_10_0', fname='model_10_0')
    # evaluate_outputs(output_folder='generations/model_9_1', fname='model_9_1')
    # evaluate_outputs(output_folder='generations/model_8_2', fname='model_8_2')
    # evaluate_outputs(output_folder='generations/model_7_3', fname='model_7_3')
    # evaluate_outputs(output_folder='generations/model_6_4', fname='model_6_4')
    # evaluate_outputs(output_folder='generations/model_5_5', fname='model_5_5')

    eval_set = 'nongendered_test_set'

    with open(f'{eval_set}/source.txt', 'r') as f:
        source = f.readlines()
    #
    female_indices = [i for i, sent in enumerate(source) if is_gendered(sent) == 'female']
    male_indices = [i for i, sent in enumerate(source) if is_gendered(sent) == 'male']
    print(len(female_indices))
    print(len(male_indices))

    algorithms = ['convert', 'old_convert_2', 'old_convert_1', 'prior_work', 'identity']
    # algorithms = ['model_5_5', 'model_6_4', 'model_7_3', 'model_8_2', 'model_9_1', 'model_10_0', 'model_full']
    for algo in algorithms:
        generation_fine_grained = split_generation(generation_folder=f"{algo}",
                                                   female_indices=female_indices,
                                                   male_indices=male_indices)

        for domain, generation in generation_fine_grained.items():
            with open(f'{eval_set}/generations/{algo}/{domain}.generation', 'w') as f:
                for sent in generation:
                    f.write(sent)

        evaluate_outputs(output_folder=f'generations/{algo}',
                         fname=algo)


if __name__ == "__main__":
    main()
