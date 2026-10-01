"""Language guard: keep non-English documents out of scoring.

Every reference distribution chaff ships — the artifact lexicons, the human baseline,
the calibration — is English. A French or Russian document compared with them is not
"atypical English"; it is outside what the instrument measures, and scoring it produces
confident nonsense (a German page has almost no English function words, which the
distributional family reads as an extreme). Such documents are reported ``UNSCORED``
with a reason instead.

The guard asks for **positive evidence of another language**, never mere absence of
English. English documents with almost no function words exist — the ASCII table, the
Unicode property index, option tables — and calling those non-English would be wrong.
So a document is non-English only when:

* fewer than :data:`MIN_LATIN_SHARE` of its letters are Latin script (Cyrillic, Greek,
  Arabic, Hebrew, Devanagari, Thai, CJK, ...), or
* function words of another Latin-script language make up at least
  :data:`MIN_FOREIGN_SHARE` of its words *and* outnumber English function words by
  :data:`FOREIGN_TO_ENGLISH`.

Single letters are ignored on both sides: ``e``, ``o``, ``y``, ``a`` are function words in
several languages and variable names in every technical document.

Measured (phase 7) on 7,480 English documents — the human baseline, MAGE, the blind
synthetic sets, Gutenberg chunks, licences — and 1,371 localized macOS help and licence
files in 40+ languages: no English document flagged, 98.7% of the non-English ones
caught. The two English-corpus documents it did flag are not English: a French review
and a Welsh news story inside MAGE. Misses are languages without a word list here
(Lithuanian) and short Polish, Slovenian and Finnish pages. The word lists were written
from general knowledge before that measurement; the two thresholds were chosen on it.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, FrozenSet, Optional, Sequence

#: Below this share of Latin-script letters a document is not English. English documents
#: measured no lower than 0.96 (non-ASCII symbols, the odd quoted name).
MIN_LATIN_SHARE = 0.8
#: Another language's function words must make up at least this share of the words...
MIN_FOREIGN_SHARE = 0.03
#: ...and outnumber English function words by this factor.
FOREIGN_TO_ENGLISH = 1.5

_ENGLISH = """the of and to in is it that was for on are as with be by at this have from or
an but not they which you he she we his her their there were been has had will would can
could should may might must shall do does did if then than so what when where who whom whose
how why all any each other some such no nor only own same too very just into about over after
before between through during under above below up down out off again further once here more
most both few these those its our your my me him them us am being having doing"""

#: The commonest function words of the Latin-script languages most likely to turn up
#: in a web crawl. Words that are also English are dropped when the sets are built.
_FOREIGN_BY_LANGUAGE = {
    "French": "le la les des du un une est et que qui dans pour pas sur avec ce cette il elle "
              "sont au aux vous nous ou mais leur ses son sa par plus être été",
    "German": "der die das und ist nicht mit den dem des ein eine einen zu von sich auf für auch "
              "es sie wir ich werden wird sind oder bei aus nach wie kann",
    "Spanish": "el los las del que en es por con para una su se lo como más pero sus al esta "
               "este son puede ser",
    "Italian": "il di che è la per un una con non sono del della le gli dei delle questo nel "
               "alla essere può anche",
    "Portuguese": "os da do das dos que é em um uma para com não se ao na no mais pode ser seu sua",
    "Dutch": "de het een en van dat op te zijn niet met voor er ook worden wordt kan aan bij "
             "uit naar deze",
    "Swedish": "och att det som en på är av för med till den inte har om ett kan de du",
    "Danish/Norwegian": "og at det som en på er af av for med til den ikke har om et kan de du",
    "Finnish": "ja on ei se että tai kun joka mutta voi ovat sinun tämä",
    "Polish": "w na nie się że do jest to jak lub oraz może są ten",
    "Czech/Slovak": "na je se sa že do nebo alebo jako ako ve vo pro pre to být byť může môže",
    "Croatian/Slovenian": "na je se da za od ili ali kao kot može lahko su so in",
    "Romanian": "și în de la cu că pe nu este sau pentru un care sunt",
    "Catalan": "el la els les de que en es per amb no un una del al són pot",
    "Hungarian": "az és hogy nem egy is van meg csak vagy ha de mint",
    "Turkish": "ve bir bu da de için ile ne değil olarak veya daha çok",
    "Indonesian/Malay": "dan yang di untuk dengan ini itu dari tidak ke atau pada dapat anda",
    "Vietnamese": "và của có là được không cho các những một này với để trong",
}

ENGLISH: FrozenSet[str] = frozenset(w for w in _ENGLISH.split() if len(w) > 1)
FOREIGN: Dict[str, FrozenSet[str]] = {
    lang: frozenset(w for w in words.split() if len(w) > 1 and w not in ENGLISH)
    for lang, words in _FOREIGN_BY_LANGUAGE.items()
}
_ALL_FOREIGN: FrozenSet[str] = frozenset().union(*FOREIGN.values())


def latin_share(words: Sequence[str]) -> float:
    """Share of letters in ``words`` that are Latin script."""
    letters = latin = 0
    for word in words:
        for ch in word:
            letters += 1
            if "LATIN" in unicodedata.name(ch, ""):
                latin += 1
    return latin / letters if letters else 1.0


def non_english_reason(words: Sequence[str]) -> Optional[str]:
    """Why a document is not English, or ``None`` when there is no positive evidence
    that it is in another language. ``words`` are lowercased word tokens."""
    if not words:
        return None
    latin = latin_share(words)
    if latin < MIN_LATIN_SHARE:
        return "not English: {0:.0%} of its letters are Latin script".format(latin)
    n = float(len(words))
    english = sum(1 for w in words if w in ENGLISH) / n
    foreign = sum(1 for w in words if w in _ALL_FOREIGN) / n
    if foreign >= MIN_FOREIGN_SHARE and foreign > FOREIGN_TO_ENGLISH * english:
        best = max(FOREIGN, key=lambda lang: sum(1 for w in words if w in FOREIGN[lang]))
        return ("not English: {0:.0%} of its words are function words of another language "
                "(best match among the known lists: {1}) against {2:.0%} English ones".format(foreign, best, english))
    return None
