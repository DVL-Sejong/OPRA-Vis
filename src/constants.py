USE_CACHE = True

CONCEPTS = {
    'opra': [
        'Trust',
        'Satisfaction',
        'Control Mutuality',
        'Commitment',
    ],
    'toxicity': [
        'Obscene',
        'Sexual Explicit',
        'Identity Attack',
        'Insult',
        'Threat',
    ],
}

NUM_CONCEPT_DIMENSION = {
    'opra': 2,
    'toxicity': 1,
}

class ConceptColumnRename:
    def __init__(self, data):
        self.data = data
        self.reversed_data = self._create_reverse_mapping(data)

    def __getitem__(self, key):
        """Get the renamed column."""
        if isinstance(key, str):
            return self.data.get(key, None)
        elif isinstance(key, tuple) and len(key) == 2:
            concept, column = key
            return self.data.get(concept, {}).get(column, None)
        else:
            raise KeyError(f'Invalid key format: {key}. Expected str or tuple of (concept, column).')

    def get_reverse(self, concept, column):
        """Get the original column name from the renamed column."""
        return self.reversed_data.get(concept, {}).get(column, None)

    def _create_reverse_mapping(self, data):
        """Create a reverse mapping for quick lookups."""
        reversed_data = {}
        for concept, mapping in data.items():
            reversed_data[concept] = {v: k for k, v in mapping.items()}
        return reversed_data
CONCEPT_COLUMN_RENAME = ConceptColumnRename({
    'opra': {
        'trust': 'Trust',
        'commitment': 'Commitment',
        'control_mutuality': 'Control Mutuality',
        'satisfaction': 'Satisfaction',
        'llm_trust': 'Trust',
        'llm_commitment': 'Commitment',
        'llm_control_mutuality': 'Control Mutuality',
        'llm_control mutuality': 'Control Mutuality',
        'llm_satisfaction': 'Satisfaction',
    },
    'toxicity': {
        'obscene': 'Obscene',
        'sexual_explicit': 'Sexual Explicit',
        'identity_attack': 'Identity Attack',
        'insult': 'Insult',
        'threat': 'Threat',
    }
})
CONCEPT_COLUMN_NAMES = {
    'opra': [
        'trust',
        'commitment',
        'control_mutuality',
        'satisfaction',
    ],
    'toxicity': [
        'Obscene',
        'Sexual Explicit',
        'Identity Attack',
        'Insult',
        'Threat',
    ]
}

DATA_TEXT_FIELD = {
    'amazon': 'sentence',
    'local': 'sentence',
    'jigsaw': 'comment_text',
}

CLUE_LABELS = ['true', 'false']

SENTIMENT_LABELS = ['pos', 'neg']
