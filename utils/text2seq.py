import numpy as np
from g2p_en import G2p
from text import *
from text.cleaners import custom_english_cleaners
from text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}
g2p = G2p()


def text2symbols(text, dtype):
    clean_char = custom_english_cleaners(text.rstrip()).rstrip()
    if clean_char[-1] in ['.', ',']:
        while clean_char[-1] in ['.', ',']:
            clean_char = clean_char[:-1]
        clean_char = clean_char + '.'
    elif clean_char[-1] in ['!', '?']:
        clean_char = clean_char
    else:
        clean_char = clean_char + '.'
    
    if dtype=='char':
        return clean_char

    clean_phone = []
    for s in g2p(clean_char.lower()):
        if (s in [',', '!', '.', '?', "'"]) and (clean_phone[-1]==' '):
            clean_phone.pop()
            clean_phone.append(s)

        elif '@'+s in symbol_to_id:
            clean_phone.append('@'+s)

        else:
            clean_phone.append(s)
    
    return clean_phone


def symbols2seq(symbols):
    return np.asarray([symbol_to_id[s] for s in symbols], dtype=np.int64)


def text2seq(text, dtype='phone'):
    symbols = text2symbols(text, dtype)
    return symbols2seq(symbols)