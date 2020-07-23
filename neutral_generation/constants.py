# pronouns
NON_FUNCTION_PRONOUNS = {   # need to change pronoun based on context
    # one-to-many mapping
    'her': ['them',     # I talked to her --> I talked to them
            'their'],   # This is her pen --> This is their pen
    'his': ['their',    # This is his pen --> This is their pen
            'theirs']   # This pen is his --> This pen is theirs
}

NON_INJECTIVE_PRONOUNS = {
    # many-to-one mapping
    'he': 'they',
    'she': 'they',
    "she's": "they're",
    "he's": "they're",
    'herself': 'themself',  # TODO: be careful with cases such as "Sarah herself" --> "Sarah themself"
    'himself': 'themself'
}

INJECTIVE_PRONOUNS = {
    # one-to-one mapping
    'him': 'them',      # I talked to him --> I talked to them
    'hers': 'theirs'    # This pen is hers --> This pen is theirs
}

EASY_PRONOUNS = NON_INJECTIVE_PRONOUNS.copy()
EASY_PRONOUNS.update(INJECTIVE_PRONOUNS)  # these pronouns can be replaced with regex

OCCUPATION_WORDS = {
    'policeman': 'police officer',
    'policewoman': 'police officer',
    'policemen': 'police officers',
    'policewomen': 'police officers',
    'stewardess': 'flight attendant',
    'weatherman': 'weather reporter',
    'fireman': 'firefighter',
    'chairman': 'chair',
    'spokesman': 'spokesperson'     # TODO: gradually add more words
}

GENDER_SPECIFIC = {
    'mankind': 'humanity',
    'layman': 'layperson',
    'laymen': 'lay people',
}

GENDERED_TERMS = OCCUPATION_WORDS
GENDERED_TERMS.update(GENDER_SPECIFIC)


# verbs
RULE_CHANGE_VERBS = {
    'is': 'are',
    'has': 'have',
    'was': 'were'
}

IRREGULAR_PRESENT_SIMPLE_VERBS = {
    'is': 'are',
    "'s": "'re",
    'has': 'have',
    'does': 'do',
    'doesn': 'don',
    'goes': 'go'
}

VERB_ES_SUFFIXES = ['ses', 'zes', 'xes', 'ches', 'shes']