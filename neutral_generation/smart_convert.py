import re
import math
import torch
import spacy

# SpaCy: lowercase is for dependency parser, uppercase is for part-of-speech tagger
from spacy.symbols import nsubj, nsubjpass, conj, poss, obj, iobj, pobj, dobj, VERB, AUX, PRON
from spacy.tokens import Token
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from constants import *

nlp = spacy.load("en_core_web_sm")

# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

# direct replacement mapping
SIMPLE_REPLACE = EASY_PRONOUNS
SIMPLE_REPLACE.update(GENDERED_TERMS)


def convert(sentence: str) -> str:
    """
    convert a sentence to gender-neutral form
    :param sentence: sentence meeting SNAPE criteria (meaning 1 entity and 1 gender)
    :return: sentence in gender-neutral form
    """

    # use a LM to break ties for pronouns when there is a one-to-many mapping
    for word, choices in NON_FUNCTION_PRONOUNS.items():
        sentence = smart_pronoun_replace(sentence, word, choices)

    # replace pronouns and gendered terms that have a clear mapping
    # cannot directly modify "doc" object, so we will instead create a "replacement" attribute
    # in the end, we will create a new doc from original doc and any replacements
    Token.set_extension("simple_replace", getter=simple_replace, force=True)

    # Doc is a SpaCy container comprised of a sequence of Token objects
    doc = nlp(sentence)

    # create a dictionary mapping verbs in sentence from third-person singular to third-person plural
    verbs_auxiliaries = identify_verbs_and_auxiliaries(doc=doc)
    verbs_replacements = pluralize_verbs(verbs_auxiliaries)

    # create a new doc with replacements for pronouns, verbs, and gendered terms
    new_sentence = create_new_doc(doc, verbs_replacements)

    return new_sentence


def smart_pronoun_replace(sentence: str, token: str, choices: list) -> str:
    """
    use an LM to choose between multiple options for a replacement (e.g. her --> their / them)
    :param sentence: input sentence
    :param token: token with more than one choice for replacement
    :param choices: the options for replacement
    :return: the sentence after the LM has chosen the replacement option with lower perplexity
    """
    # generate all choices, then choose best option using LM (one with lowest perplexity)
    sentence_scores = dict()
    for choice in choices:
        new_sentence = regex_token_replace(sentence, token, replacement=choice)
        if sentence != new_sentence:
            new_score = score(new_sentence)
            sentence_scores[new_sentence] = new_score

    if not sentence_scores:  # source pronoun not found in sentence, meaning there are no choices to choose from
        return sentence

    return min(sentence_scores, key=sentence_scores.get)  # return sentence with lowest score (perplexity)


def regex_token_replace(sentence: str, token: str, replacement: str) -> str:
    """
    replace all occurrences of a target token with its replacement
    :param sentence: input sentence to be modified
    :param token: target token to be replaced
    :param replacement: replacement word for the target token
    :return: sentence with the all occurrences of the target token substituted by its replacement
    """
    replace_map = [[token, replacement], [token.capitalize(), replacement.capitalize()],
                   [token.upper(), replacement.upper()]]

    for j in range(len(replace_map)):
        pattern = re.compile(r'\b{}\b'.format(replace_map[j][0]))  # \b indicates a word boundary in regex
        sentence = re.sub(pattern, replace_map[j][1], sentence)

    return sentence


def score(sentence: str) -> float:
    """
    score the perplexity of a sentence
    :param sentence: input sentence
    :return: perplexity normalized by length of sentence (longer sentences won't have inherently have higher perplexity)
    """
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss) / len(tokenize_input)  # normalize perplexity by number of tokens


def simple_replace(token):
    """
    mainly deals with straightforward cases of pronoun / gendered word replacement using a lookup
    also resolves "her" --> "their" / "them"
    :param token: SpaCy token
    :return: the token's text replacement (if it exists) as a string.
    """
    text = token.text

    # use dependency parser to resolve "her" --> "their" / "them"
    # if "her" is a possessive pronoun, then its replacement should be "their"
    # if "her" is an object, then its replacement should be "them"
    if text.lower() == 'her':
        is_obj = (token.dep == obj or
                  token.dep == iobj or
                  token.dep == pobj or
                  token.dep == dobj or
                  token.dep_ == "dative")
        if token.dep == poss:
            return capitalization_helper(original=text,
                                         replacement='their')
        elif is_obj:
            return capitalization_helper(original=text,
                                         replacement='them')
        else:
            return None

    # use a lookup for direct mappings
    # e.g. he --> they, she --> they, policeman --> police officer
    elif text.lower() in SIMPLE_REPLACE.keys():
        replace = SIMPLE_REPLACE[text.lower()]
        return capitalization_helper(original=text,
                                     replacement=replace)

    return None


def capitalization_helper(original: str, replacement: str) -> str:
    """
    helper function to return appropriate capitalization
    :param original: original word from the sentence
    :param replacement: replacement for the given word
    :return: replacement word matching the capitalization of the original word
    """
    # check for capitalization
    if original[0].isupper():
        return replacement.capitalize()
    elif original.isupper():
        return replacement.upper()

    # otherwise, return the default replacement
    return replacement


def identify_verbs_and_auxiliaries(doc) -> dict:
    """
    identify the root verbs and their corresponding auxiliaries with 'she' or 'he' as their subject
    :param doc: input Doc object
    :return: dictionary with verbs (SpaCy Token) as keys, auxiliaries as values (SpaCy Token)
    """
    # no need to include uppercase pronouns bc searching for potential_subject checks lower-cased version of each token
    SUBJECT_PRONOUNS = ['she', 'he']

    # identify all verbs
    verbs = set()
    # this deals with repeating verbs, e.g. "He sings and sings."
    # verb Token with same text will have different position (makes them unique)
    for possible_subject in doc:
        is_subject = (
                (possible_subject.dep == nsubj or
                 possible_subject.dep == nsubjpass) and  # current token is a subject
                # head of current token is a verb
                (possible_subject.head.pos == VERB or possible_subject.head.pos == AUX) and
                possible_subject.text.lower() in SUBJECT_PRONOUNS  # current token is either she / he
        )
        if is_subject:
            verbs.add(possible_subject.head)

    # identify all conjuncts and add them to set of verbs
    # e.g. he dances and prances --> prances would be a conjunct
    for possible_conjunct in doc:
        is_conjunct = (
                possible_conjunct.dep == conj and  # current token is a conjunct
                possible_conjunct.head in verbs  # the subject of that verb is she / he
        )
        if is_conjunct:
            verbs.add(possible_conjunct)

    verbs_auxiliaries = dict()
    for verb in verbs:
        verbs_auxiliaries[verb] = list()
    for possible_aux in doc:
        is_auxiliary = (
                possible_aux.pos == AUX and  # current token is an auxiliary verb
                possible_aux.head in verbs  # the subject of that verb is she / he
        )
        if is_auxiliary:
            verb = possible_aux.head
            verbs_auxiliaries[verb].append(possible_aux)

    return verbs_auxiliaries


def pluralize_verbs(verbs_auxiliaries: dict) -> dict:
    """
    map each verb and auxiliary to its plural form
    :param verbs_auxiliaries: dictionary with verbs (SpaCy Token) as keys, auxiliaries as values (SpaCy Token)
    :return: dictionary with verbs and auxiliaries (SpaCy Token) as keys, plural form as values (str or None)
    """
    verbs_replacements = dict()

    for verb, auxiliaries in verbs_auxiliaries.items():
        # verb has no auxiliaries
        if not auxiliaries:
            verbs_replacements[verb] = pluralize_single_verb(verb)

        # there are auxiliary verbs
        else:
            verbs_replacements[verb] = None  # do not need to pluralize root verb if there are auxiliaries

            # use a lookup to find replacements for auxiliaries
            for auxiliary in auxiliaries:
                text = auxiliary.text
                if text.lower() in IRREGULAR_VERBS.keys():
                    replacement = IRREGULAR_VERBS[text.lower()]
                    verbs_replacements[auxiliary] = capitalization_helper(original=text,
                                                                          replacement=replacement)
                else:
                    verbs_replacements[auxiliary] = None

    return verbs_replacements


def pluralize_single_verb(verb):
    """
    pluralize a single verb
    :param verb: verb as a SpaCy token
    :return: the plural form of the verb as a str, or None if verb doesn't lend itself to pluralization
    """
    verb_text = verb.text

    # check verb tense (expect to be either past simple or present simple)
    verb_tag = nlp.vocab.morphology.tag_map[verb.tag_]

    if 'Tense_past' in verb_tag.keys():
        # was is an irregular past tense verb from third-person singular to third-person plural
        if verb_text.lower() == 'was':
            return capitalization_helper(verb_text, 'were')

        # other past-tense verbs remain the same
        else:
            return None

    # oftentimes, if there are 2+ verbs in a sentence, each verb after the first (the conjuncts) will be misclassified
    # the POS of these other verbs are usually misclassified as NOUN
    # e.g. He dances and prances and sings. --> "prances" and "sings" are conjuncts marked as NOUN (should be VERB)
    # checking if verb ends with "s" is a band-aid fix
    elif 'Tense_pres' in verb_tag.keys() or verb.text.endswith('s'):
        return capitalization_helper(original=verb_text.lower(),
                                     replacement=pluralize_present_simple(verb_text))

    return None


def pluralize_present_simple(lowercase_verb: str):
    """
    pluralize a third-person singular verb in the present simple tense
    :param lowercase_verb: original verb (lower-cased)
    :return: 3rd-person plural verb in the present simple tense
    """
    # TODO: pluralizing present tense can be tricky.
    # Probably a good idea to write a script to test function, can easily get ground truth for evaluation
    for singular, plural in IRREGULAR_VERBS.items():
        if lowercase_verb == singular:
            return plural

    if lowercase_verb.endswith('ies'):
        return lowercase_verb[:-3] + 'y'

    for suffix in VERB_ES_SUFFIXES:
        if lowercase_verb.endswith(suffix):
            return lowercase_verb[:-2]

    if lowercase_verb.endswith('s'):
        return lowercase_verb[:-1]

    return None


def create_new_doc(doc, verbs_replacements: dict):
    """
    create a new SpaCy doc using the original doc and a mapping of verbs to their replacements
    :param doc: original doc with simple_replace extension (from simple_replace function)
    :param verbs_replacements: dictionary mapping verbs and auxiliaries to their replacements
    :return: the gender-neutral sentence as a SpaCy doc
    """
    token_texts = []
    for token in doc:
        replace_verb = (token in verbs_replacements.keys() and
                        verbs_replacements[token])

        if token._.simple_replace:
            token_texts.append(token._.simple_replace)

        elif replace_verb:
            token_texts.append(verbs_replacements[token])

        else:
            token_texts.append(token.text)
        if token.whitespace_:  # filter out empty strings
            token_texts.append(token.whitespace_)

    new_sentence = ''.join(token_texts)
    return new_sentence