"""The language guard: evidently non-English documents are UNSCORED, with a reason."""

from __future__ import annotations

import json
import os

import pytest

from chaff.cli import main
from chaff.language import ENGLISH, FOREIGN, non_english_reason
from chaff.pipeline import score_corpus
from chaff.tokenization import tokenize_words

ENGLISH_PROSE = (
    "The river was higher than anyone in the village could remember. By the time the "
    "morning bell rang, the water had reached the steps of the old mill, and the miller "
    "was carrying sacks of flour up to the loft with the help of his two sons. Nobody "
    "spoke much. They had seen floods before, but not one that rose so quickly or so "
    "quietly, and there was something in its silence that made them work faster than "
    "they needed to.")

# The ASCII table and option lists: English with almost no English function words, and
# full of single letters that are function words elsewhere (e, o, y, a, i).
ENGLISH_TABLE = (
    "ASCII octal hexadecimal and decimal character sets. "
    + " ".join("{0:03o} {1} {2:02x} {1} {3} {1}".format(n, chr(c), n, n)
               for n, c in enumerate(range(97, 123)) for _ in range(2)))

FRENCH = (
    "Le village se trouve au bord de la rivière, et les habitants disent que l'eau n'a "
    "jamais été aussi haute. Dans la matinée, le meunier et ses deux fils ont porté les "
    "sacs de farine au grenier pour les mettre à l'abri. Personne ne parlait beaucoup. "
    "Ils avaient déjà vu des crues, mais jamais une qui montait si vite et sans bruit, et "
    "il y avait dans ce silence quelque chose qui les poussait à travailler plus vite.")

GERMAN = (
    "Das Dorf liegt am Ufer des Flusses, und die Bewohner sagen, dass das Wasser noch nie "
    "so hoch gestanden hat. Am Morgen haben der Müller und seine beiden Söhne die Säcke "
    "mit Mehl auf den Dachboden getragen, damit sie trocken bleiben. Niemand sprach viel. "
    "Sie hatten schon Hochwasser gesehen, aber keines, das so schnell und so leise stieg, "
    "und in dieser Stille lag etwas, das sie schneller arbeiten ließ, als es nötig war.")

SPANISH = (
    "El pueblo está a la orilla del río, y los vecinos dicen que el agua nunca había "
    "llegado tan alto. Por la mañana, el molinero y sus dos hijos subieron los sacos de "
    "harina al desván para que no se mojaran. Nadie hablaba mucho. Ya habían visto "
    "crecidas, pero ninguna que subiera tan rápido y en silencio, y había algo en ese "
    "silencio que los hacía trabajar más deprisa de lo que hacía falta.")

RUSSIAN = (
    "Деревня стоит на берегу реки, и жители говорят, что вода никогда не поднималась так "
    "высоко. Утром мельник и его два сына перенесли мешки с мукой на чердак, чтобы они не "
    "намокли. Никто почти не разговаривал. Они уже видели наводнения, но ни одно не "
    "поднималось так быстро и так тихо, и в этой тишине было что-то, что заставляло их "
    "работать быстрее, чем нужно. Вечером вода начала медленно отступать от мельницы.")


def _reason(text):
    return non_english_reason(tokenize_words(text))


@pytest.mark.parametrize("text", [ENGLISH_PROSE, ENGLISH_TABLE], ids=["prose", "ascii-table"])
def test_english_is_never_held_out(text):
    assert _reason(text) is None


def test_english_quoting_another_language_is_still_english():
    assert _reason(ENGLISH_PROSE + ' The sign on the door read "fermé pour la journée".') is None




def test_a_stray_foreign_word_in_a_table_is_not_evidence():
    """Option and locale tables have almost no English function words, so a single "de"
    or "es" outnumbers them. Without a minimum share that would be "not English"."""
    table = " ".join("option{0} value{0} default{0}".format(i) for i in range(40)) + " locale de"
    assert _reason(table) is None

def test_english_discussing_french_needs_foreign_words_to_outnumber_english_ones():
    """A French phrase book in English uses plenty of French function words, but more
    English ones: that is English, and only the ratio says so."""
    text = ENGLISH_PROSE + (" In French you would say le moulin, la rivière, les sacs and des "
                            "fils, and for the miller you would use il or elle with est.") * 2
    words = tokenize_words(text)
    foreign = sum(1 for w in words if any(w in v for v in FOREIGN.values())) / len(words)
    assert foreign >= 0.03            # enough to trip the share test on its own
    assert _reason(text) is None

@pytest.mark.parametrize("text,language", [(FRENCH, "French"), (GERMAN, "German"), (SPANISH, "Spanish")])
def test_latin_script_languages_are_recognised_by_their_function_words(text, language):
    reason = _reason(text)
    assert reason is not None and "function words" in reason and language in reason


def test_other_scripts_are_recognised_by_script():
    reason = _reason(RUSSIAN)
    assert reason is not None and "Latin script" in reason


def test_no_words_is_no_evidence():
    assert non_english_reason([]) is None


def test_word_lists_are_disjoint_from_english_and_ignore_single_letters():
    for words in FOREIGN.values():
        assert not words & ENGLISH
        assert all(len(w) > 1 for w in words)
    assert all(len(w) > 1 for w in ENGLISH)


def _corpus(tmp_path):
    path = os.path.join(str(tmp_path), "mixed.jsonl")
    texts = [ENGLISH_PROSE] * 8 + [FRENCH, GERMAN, RUSSIAN]
    with open(path, "w", encoding="utf-8") as fh:
        for i, text in enumerate(texts):
            fh.write(json.dumps({"id": "d{0}".format(i), "text": text}, ensure_ascii=False) + "\n")
    return path


def test_scoring_leaves_non_english_documents_unscored_with_a_reason(tmp_path):
    out = os.path.join(str(tmp_path), "scores.jsonl")
    run = score_corpus(_corpus(tmp_path), scores_path=out)
    rows = {r["doc_id"]: r for r in map(json.loads, open(out))}
    for doc_id in ("d8", "d9", "d10"):
        assert rows[doc_id]["tier"] == "UNSCORED" and rows[doc_id]["reason"].startswith("not English")
    assert all(rows["d{0}".format(i)]["tier"] != "UNSCORED" for i in range(8))
    assert "reason" not in rows["d0"]
    assert run.meta["profile"]["n_non_english"] == 3
    assert any("3 of 11 documents are evidently not English" in n for n in run.meta["notes"])


def test_the_guard_can_be_switched_off(tmp_path):
    run = score_corpus(_corpus(tmp_path), language_guard=False)
    assert run.meta["tiers"]["UNSCORED"] == 0
    assert run.meta["profile"]["n_non_english"] == 0


def test_cli_explains_why_a_document_was_not_scored(tmp_path, capsys):
    corpus, out = _corpus(tmp_path), os.path.join(str(tmp_path), "s.jsonl")
    assert main(["score", corpus, "--out", out]) == 0
    capsys.readouterr()
    assert main(["explain", out, "d9"]) == 0
    assert "not English" in capsys.readouterr().out
    assert main(["score", corpus, "--allow-non-english"]) == 0
    assert "UNSCORED 0" in capsys.readouterr().out
