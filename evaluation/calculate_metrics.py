from evaluate_generation import *
import os
import json

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


def create_outputs(func, output_folder):
    for file_set in SOURCE_GENERATION_ANNOTATION:
        source_file = file_set[0]
        generation_file = file_set[1]

        outputs = generate_outputs(source_file=os.path.join('test_set', source_file), func=func)
        with open(f"test_set/{output_folder}/{generation_file}", 'w') as f:
            for sent in outputs:
                f.write(sent)


def evaluate_outputs(output_folder, fname):
    scores = dict()
    for file_set in SOURCE_GENERATION_ANNOTATION:
        generation_file = file_set[1]
        annotation_file = file_set[2]

        results = get_metrics(generation_file=os.path.join('test_set', output_folder, generation_file),
                              annotation_file=os.path.join('test_set', annotation_file))

        domain = generation_file.split('.')[0]
        scores[domain] = results

    with open(f"test_set/scores/{fname}.json", 'w') as f:
        json.dump(scores, f)


def main():
    # from jewang_neutral_converter import jewang_convert
    # create_outputs(func=jewang_convert, output_folder='generations/jewang_convert')

    # from old_smart_convert import old_convert
    # create_outputs(func=old_convert, output_folder='generations/old_convert_1')

    # from old_score_smart_convert import convert_old_score
    # create_outputs(func=convert_old_score, output_folder='generations/old_convert_2')

    # from smart_convert import convert
    # create_outputs(func=convert, output_folder='generations/convert')

    evaluate_outputs(output_folder='generations/jewang_convert', fname='jewang_convert')
    evaluate_outputs(output_folder='generations/old_convert_1', fname='old_convert_1')
    evaluate_outputs(output_folder='generations/old_convert_2', fname='old_convert_2')
    evaluate_outputs(output_folder='generations/convert', fname='convert')


if __name__ == "__main__":
    main()