import re


PRONOUNS = [
  ["she's", "they're"],
  ["he's", "they're"],
  ['he', 'they'], 
  ['she', 'they'],
  ['him', 'them'],
  ['hers', 'theirs'],
  ['her', 'them'],
  ['herself', 'themselves'],
  ['himself', 'themselves'],
  ['his', 'their'],
]

IRREGULAR_VERBS = [
    ['was', 'were'],
    ['has', 'have'],
    ['is', 'are'],
    ['does', 'do'],
    ['doesn', 'don'],
    ['hasn', 'haven'],
    ['isn', 'aren'],
    ['goes', 'go'],
]

VERB_ES_SUFFIXES = ['ses', 'zes', 'xes', 'ches', 'shes']


def convert(s):
    s = pluralize_verbs(s)
    for i in range(len(PRONOUNS)):
        s = pronoun_replace(s, PRONOUNS[i][0], PRONOUNS[i][1])

    return s


def pluralize_verbs(s):
    # replaces all verbs following he / she with their plural version
    pattern = re.compile(r'''(?x)       # verbose option
                        \bs?he\b        # she or he at a word boundary
                        \s+             # 1 or more whitespaces
                        (.+?)\b         # any number of characters, non-greedy search
                        ''', re.IGNORECASE)

    matches = re.finditer(pattern, s)
    for match in matches:   # iterate through all matches with pattern
        phrase = match.group(0)
        verb = match.group(1)
        replace = phrase.replace(verb, pluralize_verb(verb))

        s = s.replace(phrase, replace)

    return s


def pluralize_verb(verb):
    uppercase = verb == verb.upper()
    new_verb = pluralize_lowercase_verb(verb.lower())
    if uppercase:
        new_verb = new_verb.upper()

    return new_verb


def pluralize_lowercase_verb(verb):
    for i in range(len(IRREGULAR_VERBS)):
        if verb == IRREGULAR_VERBS[i][0]:
            return IRREGULAR_VERBS[i][1]

    if verb.endswith('ies'):
        return verb[:-3] + 'y'

    for i in range(len(VERB_ES_SUFFIXES)):
        suffix = VERB_ES_SUFFIXES[i]
        if verb.endswith(suffix):
            return verb[:-2]

    if verb.endswith('s'):
        return verb[:-1]

    return verb


def pronoun_replace(s, p, r):
    replace_map = [[p, r], [capitalize(p), capitalize(r)], [p.upper(), r.upper()]]

    for j in range(len(replace_map)):
        pattern = re.compile(r'\b{}\b'.format(replace_map[j][0]))
        s = re.sub(pattern, replace_map[j][1], s)

    return s


def capitalize(word):
    return word[0].upper() + word[1:]


if __name__ == '__main__':
    text = '''As he enters the room, he announces that his favorite food is hot sauce.
    When he was younger, he disliked spicy food.
    Now all that drives him is his desire to acquire more hot sauce.
    People say that he's a hot sauce maniac.
    He flies around the world seeking new recipes.
    He marches to the beat of his own drum.
    He doesn't care about others' perception of him.
    SOMETIMES HE LIKES TO WRITE IN ALL CAPS.
    His wife is sometimes concerned about him.'''

    neutral_text = convert(text)
    print(neutral_text)
