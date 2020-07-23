import re
import math
import torch
import spacy

# SpaCy: lowercase is for dependency parser, uppercase is for part-of-speech tagger
from spacy.symbols import nsubj, aux, conj, neg, VERB, AUX
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from constants import *

nlp = spacy.load("en_core_web_sm")

# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


def convert(s):
    # input: SNAPE sentence (meaning 1 entity and 1 gender)
    # output: sentence in gender-neutral form

    # pluralize verbs: change all verbs in third-person singular to third-person plural
    auxiliaries = identify_verbs_and_auxiliaries(s)
    s = pluralize_verbs(s, auxiliaries)

    # use regex to replace pronouns and gendered terms that have a clear mapping
    SIMPLE_REPLACE = EASY_PRONOUNS
    SIMPLE_REPLACE.update(GENDERED_TERMS)
    for p, r in SIMPLE_REPLACE.items():
        s = regex_token_replace(s, p, r)

    # use a LM to break ties for pronouns when there is a one-to-many mapping
    for p, choices in NON_FUNCTION_PRONOUNS.items():
        s = smart_pronoun_replace(s, p, choices)

    return s


def regex_token_replace(s, old, new):
    # replace occurrences of a token in a string with a new token
    # TODO: can look into redoing replace_map with SpaCy (they save capitalization as well)
    replace_map = [[old, new], [capitalize(old), capitalize(new)], [old.upper(), new.upper()]]

    for j in range(len(replace_map)):
        pattern = re.compile(r'\b{}\b'.format(replace_map[j][0]))
        # TODO: add an exception to regex replacement when an apostrophe follows the pronoun
        # e.g. He'shan is a location, but with this rule it would become They'shan
        s = re.sub(pattern, replace_map[j][1], s)

    return s


def capitalize(word):
    return word[0].upper() + word[1:]


def smart_pronoun_replace(s, p, choices):
    # generate all choices, then choose best option using LM (one with lowest perplexity)
    # TODO: currently uses OpenAI GPT, can try other LMs to see how performance changes
    sentence_scores = dict()
    for choice in choices:
        new_sentence = regex_token_replace(s, p, new=choice)
        if s != new_sentence:
            new_score = score(new_sentence)
            sentence_scores[new_sentence] = new_score

    if not sentence_scores:   # source pronoun not found in sentence, meaning there are no choices to choose from
        return s

    print(sentence_scores)

    return min(sentence_scores, key=sentence_scores.get)    # return sentence with lowest score (perplexity)


def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss) / len(tokenize_input)     # normalize perplexity by number of tokens


def identify_verbs_and_auxiliaries(s):
    """
    :param s: input sentence
    :return: dictionary of verbs and list of auxiliaries (if any) with 'she' or 'he' as their subject
    """
    SUBJECT_PRONOUNS = ['she', 'he']
    doc = nlp(s)

    # identify all verbs
    verbs = set()
    # this deals with repeating verbs, e.g. "He sings and sings."
    #   because verb with same text will have different position (makes them unique)
    for possible_subject in doc:
        is_subject = (
            possible_subject.dep == nsubj and                   # current token is a subject
            # head of current token is a verb
            (possible_subject.head.pos == VERB or possible_subject.head.pos == AUX) and
            possible_subject.text.lower() in SUBJECT_PRONOUNS   # current token is either she / he
        )
        if is_subject:
            verbs.add(possible_subject.head)

    # identify all conjuncts and add them to set of verbs
    # e.g. he dances and prances --> prances would be a conjunct
    # TODO: understand all English labels of dependency parse better
    for possible_conjunct in doc:
        is_conjunct = (
            possible_conjunct.dep == conj and           # current token is a conjunct
            # removing the line below, can have a conjunct with an auxiliary verb, e.g. "He was angry and ran away."
            # possible_conjunct.head.pos == VERB and      # head of current token is a verb
            possible_conjunct.head in verbs             # the subject of that verb is she / he
        )
        if is_conjunct:
            verbs.add(possible_conjunct)

    # print('verbs: ', verbs)

    auxiliaries = dict()
    for verb in verbs:
        auxiliaries[verb] = list()
    for possible_aux in doc:
        is_auxiliary = (
            # current token is an auxiliary verb or negation (not)
            (possible_aux.dep == aux or possible_aux.dep == neg) and
            possible_aux.head.pos == VERB and       # head of current token is a verb
            possible_aux.head in verbs              # the subject of that verb is she / he
        )
        if is_auxiliary:
            verb = possible_aux.head
            auxiliaries[verb].append(possible_aux)

    print('verbs: ', auxiliaries)

    return auxiliaries


def pluralize_verbs(s, auxiliaries):
    """
    :param s: input sentence
    :param auxiliaries: dictionary of verbs and list of auxiliaries (if any) to be pluralized
    :return: sentence with plural form of verbs
    """

    # do_not_change includes 7 tenses in total:
    #   the 4 future tenses, the 2 past perfect tenses, and the past simple
    do_not_change = ['past simple', 'past perfect', 'future']

    # rule_change includes 4 tenses in total:
    #   the present continuous, the 2 present perfect tenses, and the past continuous
    rule_change = ['present continuous', 'present perfect', 'past continuous']

    # TODO: categorize tense for "do" verbs
    verb_tense = dict()
    for verb, auxiliary_list in auxiliaries.items():
        auxiliary_text = ' '.join(auxiliary.text for auxiliary in auxiliary_list)
        full_verb = auxiliary_text + ' ' + verb.text
        tense = categorize_tense(full_verb, verb, auxiliary_list)

        if tense == 'unknown':
            print(verb, auxiliary_list, ': unknown')

        verb_tense[verb] = tense

    # using regex to replace verbs
    # taking advantage of the idea that in SNAPE sentences, all verbs should refer to the same subject
    for verb, tense in verb_tense.items():
        if tense in do_not_change:
            # to my knowledge, the only irregular case is "was" to "were" in past simple
            # all other tenses in do_not_change would not require a modification to the verb
            if tense == 'past simple' and verb.text == 'was':
                s = regex_token_replace(s, 'was', 'were')
            continue
        elif tense in rule_change:
            for singular, plural in RULE_CHANGE_VERBS.items():
                s = regex_token_replace(s, singular, plural)
        elif tense == 'present simple':
            new_verb = pluralize_present_simple(verb.text)
            s = regex_token_replace(s, verb.text, new_verb)

    return s


def categorize_tense(full_verb, verb, auxiliary_list):
    """
    :param full_verb: auxiliary verbs + root verb as a string
    :param verb: root verb as a SpaCy token
    :return: verb tense
    """
    tokens = full_verb.split(' ')
    num_tokens = len(tokens)

    # present tenses that are easy to categorize with rule-based methods
    is_present_continuous = (
        'is' in full_verb and
        tokens[-1].endswith('ing') and
        num_tokens >= 2
    )
    if is_present_continuous:
        return 'present continuous'

    is_present_perfect = (
        'has' in full_verb and
        num_tokens >= 2
    )
    if is_present_perfect:
        return 'present perfect'

    # past tenses that are easy to categorize with rule-based methods
    is_past_continuous = (
        'was' in full_verb and
        tokens[-1].endswith('ing') and
        num_tokens >= 2
    )
    if is_past_continuous:
        return 'past continuous'

    is_past_perfect = (
        'had' in full_verb and
        num_tokens >= 2
    )
    if is_past_perfect:
        return 'past perfect'

    # all forms of future tense is easy to categorize with rule-based methods
    is_future = (
        ('will' in full_verb or 'shall' in full_verb) and
        num_tokens >= 2
    )
    if is_future:
        return 'future'

    # if verb tense is not any of the above, categorize as either present simple or past simple
    # check root verb
    verb_tag = nlp.vocab.morphology.tag_map[verb.tag_]
    # TODO: understand verb tag better
    if 'Tense_pres' in verb_tag.keys():
        return 'present simple'

    if 'Tense_past' in verb_tag.keys():
        return 'past simple'

    # check auxiliary verbs
    for auxiliary in auxiliary_list:
        aux_tag = nlp.vocab.morphology.tag_map[auxiliary.tag_]

        if 'Tense_pres' in aux_tag.keys():
            return 'present simple'
        if 'Tense_past' in aux_tag.keys():
            return 'past simple'

    return 'unknown'


def pluralize_present_simple(verb):
    uppercase = verb == verb.upper()
    new_verb = pluralize_lowercase_verb(verb.lower())
    if uppercase:
        new_verb = new_verb.upper()

    return new_verb


def pluralize_lowercase_verb(verb):
    # TODO: pluralizing present tense can be tricky.
    #   Probably a good idea to write a script to test function, can easily get ground truth for evaluation
    for singular, plural in IRREGULAR_PRESENT_SIMPLE_VERBS.items():
        if verb == singular:
            return plural

    if verb.endswith('ies'):
        return verb[:-3] + 'y'

    for i in range(len(VERB_ES_SUFFIXES)):
        suffix = VERB_ES_SUFFIXES[i]
        if verb.endswith(suffix):
            return verb[:-2]

    if verb.endswith('s'):
        return verb[:-1]

    return verb