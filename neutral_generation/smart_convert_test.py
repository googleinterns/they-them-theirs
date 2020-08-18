# using pytest to test functions in smart_convert.py
import pytest

import torch
import math

from constants import *

# direct replacement mapping
SIMPLE_REPLACE = EASY_PRONOUNS
SIMPLE_REPLACE.update(GENDERED_TERMS)

# load SpaCy's "en_core_web_sm" model
# English multi-task CNN trained on OntoNotes
# Assigns context-specific token vectors, POS tags, dependency parse and named entities
# https://spacy.io/models/en
import en_core_web_sm

nlp = en_core_web_sm.load()

# Load pre-trained language model and tokenizer
# https://huggingface.co/transformers/model_doc/gpt2.html
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model_id = 'gpt2'  # can change to gpt2-large if speed is not an issue
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

from smart_convert import convert, smart_pronoun_replace, regex_token_replace, score, simple_replace, \
    capitalization_helper, identify_verbs_and_auxiliaries, pluralize_verbs, pluralize_single_verb, \
    pluralize_present_simple


class TestConvert:
    # test convert function to rewrite gendered sentences to be gender-neutral

    def test_identity_1(self):
        assert convert("They are going to the park.") == "They are going to the park."

    def test_identity_2(self):
        assert convert("He'shan") == "He'shan"

    def test_identity_3(self):
        assert convert("the the the the the") == "the the the the the"

    def test_convert_his_1(self):
        assert convert("This is his pen.") == "This is their pen."

    def test_convert_his_2(self):
        assert convert("This pen is his.") == "This pen is theirs."

    def test_convert_her_1(self):
        assert convert("This is her pen.") == "This is their pen."

    def test_convert_her_2(self):
        assert convert("This pen belongs to her.") == "This pen belongs to them."


class TestSmartPronounReplace:
    # test smart_pronoun_replace function to replace pronouns in context

    def test_smart_pronoun_replace_shes_1(self):
        assert smart_pronoun_replace(sentence="She's going to the mall.",
                                     token="she's",
                                     choices=NON_FUNCTION_PRONOUNS["she's"]) == \
               "They're going to the mall."

    def test_smart_pronoun_replace_shes_2(self):
        assert smart_pronoun_replace(sentence="She's been feeling ill.",
                                     token="she's",
                                     choices=NON_FUNCTION_PRONOUNS["she's"]) == \
               "They've been feeling ill."

    def test_smart_pronoun_replace_hes_1(self):
        assert smart_pronoun_replace(sentence="He's going to the mall.",
                                     token="he's",
                                     choices=NON_FUNCTION_PRONOUNS["he's"]) == \
               "They're going to the mall."

    def test_smart_pronoun_replace_hes_2(self):
        assert smart_pronoun_replace(sentence="He's been feeling ill.",
                                     token="he's",
                                     choices=NON_FUNCTION_PRONOUNS["he's"]) == \
               "They've been feeling ill."


class TestRegexTokenReplace:
    # test regex_token_replace to replace tokens in a string with regular expressions

    # cases where one instance of the token appears in the sentence
    def test_replace_one_token_1(self):
        assert regex_token_replace(sentence="She is a baller.",
                                   token='she',
                                   replacement=SIMPLE_REPLACE['she']) == \
               "They is a baller."

    def test_replace_one_token_2(self):
        assert regex_token_replace(sentence="He is a baller.",
                                   token='he',
                                   replacement=SIMPLE_REPLACE['he']) == \
               "They is a baller."

    def test_replace_one_token_3(self):
        assert regex_token_replace(sentence="The kitty was saved by the fireman.",
                                   token='fireman',
                                   replacement=SIMPLE_REPLACE['fireman']) == \
               "The kitty was saved by the firefighter."

    def test_replace_one_token_4(self):
        assert regex_token_replace(sentence="The policeman was a sergeant.",
                                   token='policeman',
                                   replacement=SIMPLE_REPLACE['policeman']) == \
               "The police officer was a sergeant."

    # cases where multiple instances of the token appear in the sentence
    def test_replace_multiple_tokens_1(self):
        assert regex_token_replace(sentence="She was tired because she had the flu.",
                                   token='she',
                                   replacement=SIMPLE_REPLACE['she']) == \
               "They was tired because they had the flu."

    def test_replace_multiple_tokens_2(self):
        assert regex_token_replace(sentence="He was tired because he had the flu.",
                                   token='he',
                                   replacement=SIMPLE_REPLACE['he']) == \
               "They was tired because they had the flu."

    def test_replace_multiple_tokens_3(self):
        assert regex_token_replace(sentence="Stewardess stewardess stewardess.",
                                   token='stewardess',
                                   replacement=SIMPLE_REPLACE['stewardess']) == \
               "Flight attendant flight attendant flight attendant."

    # check for replacement while maintaining capitalization
    def test_replace_token_capitalization_1(self):
        assert regex_token_replace(sentence="MANKIND",
                                   token='mankind',
                                   replacement=SIMPLE_REPLACE['mankind']) == \
               "HUMANITY"

    def test_replace_token_capitalization_2(self):
        assert regex_token_replace(sentence="Weatherman",
                                   token='weatherman',
                                   replacement=SIMPLE_REPLACE['weatherman']) == \
               "Weather reporter"

    def test_replace_token_capitalization_3(self):
        assert regex_token_replace(sentence="chairman",
                                   token='chairman',
                                   replacement=SIMPLE_REPLACE['chairman']) == \
               "chair"


class TestScore:
    # test score function to evaluate sentence perplexity

    # use math.isclose to account for floating point error
    # these calculations are specific to the GPT-2 base language model
    def test_perplexity_calculation_1(self):
        if model_id == 'gpt2':
            assert math.isclose(score("He went to the library.", stride=1),
                                math.exp((6.5366 + 1.3047 + 1.5648 + 4.9886 + 2.7016) / 5), rel_tol=0.0001)

    # without ending period
    def test_perplexity_calculation_2(self):
        if model_id == 'gpt2':
            assert math.isclose(score("He went to the library", stride=1),
                                math.exp((6.5366 + 1.3047 + 1.5648 + 4.9886) / 4), rel_tol=0.0001)

    def test_perplexity_calculation_3(self):
        if model_id == 'gpt2':
            assert math.isclose(score("She was a kind person.", stride=1),
                                math.exp((3.6758 + 2.7864 + 6.0170 + 2.7715 + 1.8178) / 5), rel_tol=0.0001)

    def test_perplexity_calculation_4(self):
        if model_id == 'gpt2':
            assert math.isclose(score("Do they know what happened to John?", stride=1),
                                math.exp((5.4927 + 2.9635 + 1.7828 + 4.1989 + 1.2399 + 6.6663 + 2.0147) / 7),
                                rel_tol=0.0001)

    def test_perplexity_calculation_5(self):
        if model_id == 'gpt2':
            assert math.isclose(score("the the the the the the", stride=1),
                                math.exp((4.6442 + 6.6947 + 3.8328 + 1.4247 + 1.1237) / 5), rel_tol=0.001)

    # lower perplexity is better
    # lower perplexity means model has more confidence about its generations --> more likely for sentence to be coherent
    def test_common_sense_perplexity_1(self):
        assert score("The exams were due at noon.", stride=1) < score("The exams were due at exam.", stride=1)

    def test_common_sense_perplexity_2(self):
        assert score("I decided that I wanted ice cream.", stride=1) < score("I decided that I wanted shelf.", stride=1)

    def test_common_sense_perplexity_3(self):
        assert score("The grass was green.", stride=1) < score("The grass was red.", stride=1)


class TestSimpleReplace:
    # test simple_replace for straightforward cases of token replacement

    doc_1 = nlp("He helped her with her homework.")

    # test straightforward cases for replacement
    def test_simple_replace_1(self):
        # He --> They
        assert simple_replace(token=self.doc_1[0]) == 'They'

    def test_simple_replace_2(self):
        # helped --> None
        assert simple_replace(token=self.doc_1[1]) is None

    def test_simple_replace_3(self):
        # with --> None
        assert simple_replace(token=self.doc_1[3]) is None

    def test_simple_replace_4(self):
        # homework --> None
        assert simple_replace(token=self.doc_1[5]) is None

    # replacing the token 'her' in context using the dependency parser
    def test_simple_replace_5(self):
        # . --> None
        assert simple_replace(token=self.doc_1[6]) is None

    def test_replace_her_1(self):
        # her --> them
        assert simple_replace(token=self.doc_1[2]) == 'them'

    def test_replace_her_2(self):
        # her --> their
        assert simple_replace(token=self.doc_1[4]) == 'their'


class TestCapitalizationHelper:
    # test capitalization_helper to return appropriate capitalization of a token's replacement

    def test_capitalization_helper_1(self):
        assert capitalization_helper(original="him", replacement="them") == "them"

    def test_capitalization_helper_2(self):
        assert capitalization_helper(original="HERS", replacement="them") == "THEM"

    def test_capitalization_helper_3(self):
        assert capitalization_helper(original="She", replacement="they") == "They"

    def test_capitalization_helper_4(self):
        assert capitalization_helper(original="policewomen", replacement="police officers") == "police officers"

    def test_capitalization_helper_5(self):
        assert capitalization_helper(original="uPaNdDoWn", replacement="downandup") == "downandup"

    def test_capitalization_helper_6(self):
        assert capitalization_helper(original="123123", replacement="456456") == "456456"


class TestVerbs:
    # test identify_verbs_and_auxiliaries, pluralize_verbs
    # working with SpaCy docs and tokens

    # test identify_verbs_and_auxiliaries to find verbs and auxiliaries with "he" / "her" as the subject
    # test pluralize_verbs to convert third-person singular verbs to be third-person plural

    # testing cases where there is one root verb and one corresponding auxiliary verb
    doc_1 = nlp("Is she going to figure it out?")

    def test_identify_verbs_and_auxiliaries_1(self):
        # {going: [Is]}
        assert identify_verbs_and_auxiliaries(doc=self.doc_1) == \
               {self.doc_1[2]: [self.doc_1[0]]}

    def test_pluralize_verbs_1(self):
        assert pluralize_verbs({self.doc_1[0]: [self.doc_1[0]]}) == \
               {self.doc_1[0]: 'Are'}

    doc_2 = nlp("Is he going to figure it out?")

    def test_identify_verbs_and_auxiliaries_2(self):
        # {going: [Is]}
        assert identify_verbs_and_auxiliaries(doc=self.doc_2) == \
               {self.doc_2[2]: [self.doc_2[0]]}

    def test_pluralize_verbs_2(self):
        assert pluralize_verbs({self.doc_2[0]: [self.doc_2[0]]}) == \
               {self.doc_2[0]: 'Are'}

    doc_3 = nlp("Are they going to figure it out?")

    def test_identify_verbs_and_auxiliaries_3(self):
        assert identify_verbs_and_auxiliaries(doc=self.doc_3) == {}

    def test_pluralize_verbs_3(self):
        assert pluralize_verbs({}) == {}

    # multiple present-tense verbs
    doc_4 = nlp("She dances and sings and twirls.")

    def test_identify_multiple_present_tense_verbs_1(self):
        # {dances: [], twirls: [], sings: []}
        assert identify_verbs_and_auxiliaries(doc=self.doc_4) == \
               {self.doc_4[1]: [],
                self.doc_4[3]: [],
                self.doc_4[5]: []}

    def test_pluralize_verbs_4(self):
        assert pluralize_verbs({self.doc_4[1]: [],
                                self.doc_4[3]: [],
                                self.doc_4[5]: []}) == \
               {self.doc_4[1]: 'dance',
                self.doc_4[3]: 'sing',
                self.doc_4[5]: 'twirl'}

    doc_5 = nlp("He dances and sings and twirls.")

    def test_identify_multiple_present_tense_verbs_2(self):
        # {dances: [], twirls: [], sings: []}
        assert identify_verbs_and_auxiliaries(doc=self.doc_5) == \
               {self.doc_5[1]: [],
                self.doc_5[3]: [],
                self.doc_5[5]: []}

    def test_pluralize_verbs_5(self):
        assert pluralize_verbs({self.doc_5[1]: [],
                                self.doc_5[3]: [],
                                self.doc_5[5]: []}) == \
               {self.doc_5[1]: 'dance',
                self.doc_5[3]: 'sing',
                self.doc_5[5]: 'twirl'}

    doc_6 = nlp("They dance and sing and twirl.")

    def test_identify_multiple_present_tense_verbs_3(self):
        assert identify_verbs_and_auxiliaries(doc=self.doc_6) == {}


class TestPluralizeSingleVerb:
    # test pluralize_single_verb to find the plural form of a verb (no auxiliaries)

    # test "was", the third-person singular conjugation of "to be" in the past tense
    doc_1 = nlp("She was happy.")

    def test_pluralize_was_1(self):
        assert pluralize_single_verb(self.doc_1[1]) == 'were'

    doc_2 = nlp("Was he happy?")

    def test_pluralize_was_2(self):
        assert pluralize_single_verb(self.doc_2[0]) == 'Were'

    doc_3 = nlp("was was was was was")

    def test_pluralize_was_3(self):
        assert pluralize_single_verb(self.doc_3[4]) == 'were'

    doc_4 = nlp("He was about to do it.")

    def test_pluralize_was_4(self):
        assert pluralize_single_verb(self.doc_4[1]) == 'were'

    # test present tense verbs
    doc_5 = nlp("She walks her dog every day.")

    def test_pluralize_present_verb_1(self):
        assert pluralize_single_verb(self.doc_5[1]) == 'walk'

    doc_6 = nlp("He teaches at the local elementary school.")

    def test_pluralize_present_verb_2(self):
        assert pluralize_single_verb(self.doc_6[1]) == 'teach'

    doc_7 = nlp("She really tries hard to succeed.")

    def test_pluralize_present_verb_3(self):
        assert pluralize_single_verb(self.doc_7[2]) == 'try'

    # test past tense verbs (pluralization is the same, so we return None)
    doc_8 = nlp("She walked her dog yesterday.")

    def test_pluralize_past_verb_1(self):
        assert pluralize_single_verb(self.doc_8[1]) is None

    doc_9 = nlp("He taught at the local elementary school.")

    def test_pluralize_past_verb_2(self):
        assert pluralize_single_verb(self.doc_9[1]) is None

    doc_10 = nlp("He really tried hard to succeed.")

    def test_pluralize_past_verb_3(self):
        assert pluralize_single_verb(self.doc_10[2]) is None


class TestPluralizePresentSimple:
    # test pluralize_present_simple to pluralize verbs (lower-cased) in the present simple tense

    # test irregular verbs
    def test_pluralize_irregular_1(self):
        assert pluralize_present_simple('is') == 'are'

    def test_pluralize_irregular_2(self):
        assert pluralize_present_simple('has') == 'have'

    def test_pluralize_irregular_3(self):
        assert pluralize_present_simple('goes') == 'go'

    def test_pluralize_irregular_4(self):
        assert pluralize_present_simple('does') == 'do'

    # test verbs that end in 's'
    def test_pluralize_s_ending_1(self):
        assert pluralize_present_simple('knows') == 'know'

    def test_pluralize_s_ending_2(self):
        assert pluralize_present_simple('pays') == 'pay'

    def test_pluralize_s_ending_3(self):
        assert pluralize_present_simple('springs') == 'spring'

    def test_pluralize_s_ending_4(self):
        assert pluralize_present_simple('writes') == 'write'

    def test_pluralize_s_ending_5(self):
        assert pluralize_present_simple('becomes') == 'become'

    def test_pluralize_s_ending_6(self):
        assert pluralize_present_simple('rides') == 'ride'

    # test verbs that end in 'es'
    def test_pluralize_es_ending_1(self):
        assert pluralize_present_simple('catches') == 'catch'

    def test_pluralize_es_ending_2(self):
        assert pluralize_present_simple('washes') == 'wash'

    def test_pluralize_es_ending_3(self):
        assert pluralize_present_simple('boxes') == 'box'

    def test_pluralize_es_ending_4(self):
        assert pluralize_present_simple('matches') == 'match'

    def test_pluralize_es_ending_5(self):
        assert pluralize_present_simple('quizzes') == 'quiz'

    def test_pluralize_es_ending_6(self):
        assert pluralize_present_simple('buses') == 'bus'

    def test_pluralize_es_ending_7(self):
        assert pluralize_present_simple('waltzes') == 'waltz'

    def test_pluralize_es_ending_8(self):
        assert pluralize_present_simple('buzzes') == 'buzz'

    # test verbs that end in 'ies'
    def test_pluralize_ies_ending_1(self):
        assert pluralize_present_simple('flies') == 'fly'

    def test_pluralize_ies_ending_2(self):
        assert pluralize_present_simple('tries') == 'try'

    def test_pluralize_ies_ending_3(self):
        assert pluralize_present_simple('studies') == 'study'

    def test_pluralize_ies_ending_4(self):
        assert pluralize_present_simple('copies') == 'copy'