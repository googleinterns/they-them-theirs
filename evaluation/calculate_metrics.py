from evaluate_generation import *
import os

FUNC_FNAME = {
    'convert': 'generation_gpt_large',
    'convert_old_score': 'generation',
    'old_convert': 'old_generation',
    'jewang_convert': 'jewang_generation',
}

SOURCE_GENERATION_FILES = {
    'source.txt': 'generation.txt',
    'domains/jokes_source.txt': 'jokes_target.txt',
    'domains/movie_quotes_source.txt': 'movie_quotes_target.txt',
    'domains/news_articles_source.txt': 'news_articles_target.txt',
    'domains/reddit_source.txt': 'reddit_target.txt',
    'domains/twitter_source.txt': 'twitter_target.txt',
    'genders/female_source.txt': 'female_target.txt',
    'genders/male_source.txt': 'male_target.txt'
}


def create_output_files(func_list):

    with open('./challenge_set/source.txt', 'r') as f:
        source = f.readlines()

    for func in func_list:
        output = [func(sentence) for sentence in source]

        with open(f'./challenge_set/{FUNC_FNAME[func.__name__]}.txt', 'w') as f:
            for sentence in output:
                f.write(sentence)


def create_outputs(func, output_folder):
    for source_file, generation_file in SOURCE_GENERATION_FILES.items():
        outputs = generate_outputs(source_file=os.path.join('test_set', source_file), func=func)
        with open(f"test_set/{output_folder}/{generation_file}", 'w') as f:
            for sent in outputs:
                f.write(sent)


def main():
    # from jewang_neutral_converter import jewang_convert
    # create_outputs(func=jewang_convert, output_folder='generations/jewang_convert')

    # from old_smart_convert import old_convert
    # create_outputs(func=old_convert, output_folder='generations/old_convert_1')

    from old_score_smart_convert import convert_old_score
    create_outputs(func=convert_old_score, output_folder='generations/old_convert_2')

    from smart_convert import convert
    create_outputs(func=convert, output_folder='generations/convert')


if __name__ == "__main__":
    main()