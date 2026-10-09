"""Language-agnostic text primitives. No gazetteers or hand-written geography: anything
country-specific (state variants, abbreviations, native-script words) is learned from the
training pairs in lexicon.py."""
import re
import unicodedata

from unidecode import unidecode

_LATIN_MARKS = re.compile(r"[̀-ͯ]")          # accents; Indic vowel signs are kept
_SPLIT = re.compile(r"[^\wऀ-෿]+")             # \w misses Indic vowel signs -> add block
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
_OCR = str.maketrans("01345678", "oleasgtb")           # digit look-alikes inside words
PLACEHOLDERS = frozenset({"null", "nan", "none"})       # literal missing-value markers in the data


_MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "ï¿½")
_BROKEN_APOS = re.compile(r"(?<=[A-Za-z])â(?=s\b)")    # "Youngâs" -> "Young's"


def fix_mojibake(s: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1 ('Ã©' -> 'é'); drop leftover debris."""
    if not s or not any(m in s for m in _MOJIBAKE_MARKERS + ("â",)):
        return s
    if any(m in s for m in _MOJIBAKE_MARKERS):
        try:
            s = s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    s = _BROKEN_APOS.sub("'", s)
    return s.replace("ï¿½", " ").replace("�", " ").replace("Â", " ")


_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")   # joiners split Indic words otherwise
_URL = re.compile(r"(?i)\b(?:https?://)?www\.\S+")
_ALIAS = re.compile(r"(?i)\b(?:d\s*/\s*b\s*/\s*a|dba|a\s*/\s*k\s*/\s*a|aka|f\s*/\s*k\s*/\s*a|fka|formerly|t\s*/\s*a)\b\.?")
_BRACKETS = re.compile(r"[\(\[\{]([^\)\]\}]*)[\)\]\}]")
_DIGITS = re.compile(r"\d+")


def fold_tokens(s: str) -> list:
    """NFKD, strip Latin accents, lowercase, split on non-word characters."""
    if not s:
        return []
    s = _ZERO_WIDTH.sub("", fix_mojibake(s))
    s = _LATIN_MARKS.sub("", unicodedata.normalize("NFKD", s)).lower().replace("_", " ")
    return [t for t in _SPLIT.split(s) if t]


def merge_letters(tokens: list) -> list:
    """Runs of single Latin letters are one abbreviation: 's c i' -> 'sci', 'l l c' -> 'llc'."""
    out, run = [], []
    for t in tokens + [""]:
        if len(t) == 1 and t.isascii() and t.isalpha():
            run.append(t)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if t:
            out.append(t)
    return out


def strip_urls(name: str) -> str:
    """Drop 'www.…' parts from names that also contain other words."""
    rest = _URL.sub(" ", name)
    return rest if rest.strip(" |,-") else name


def split_aliases(name: str) -> list:
    """Alternative names: parts around dba/aka/formerly/t/a markers and bracketed text.
    'Gildevoarc formerly R+ Home LLC' -> ['Gildevoarc', 'R+ Home LLC'];
    'Bright Nails [Corporation]' -> ['Bright Nails', 'Corporation'].  [] when there are none."""
    alts = [m.group(1) for m in _BRACKETS.finditer(name)]
    main = _BRACKETS.sub(" ", name)
    parts = [p for p in _ALIAS.split(main) if p.strip()]
    if len(parts) > 1 or alts:
        alts += parts if len(parts) > 1 else [main]
    return [a.strip() for a in alts if a.strip()]


def number_parts(t: str) -> list:
    """Digits inside a token, zero-stripped: '1c' -> ['1'], '11e' -> ['11'], 'b1409' -> ['1409']."""
    return [d.lstrip("0") or "0" for d in _DIGITS.findall(t)]


def strip_domain(name: str) -> str:
    """'www.foo-bar.com' -> 'foo-bar' (names given as web domains)."""
    if " " not in name.strip() and "." in name:
        name = re.sub(r"^(?i:www)\.", "", name.strip())
        head = name.rsplit(".", 1)[0]
        return head or name
    return name


def is_ascii(t: str) -> bool:
    return t.isascii()


def apply_phrases(tokens: list, lex: dict) -> list:
    """Replace learned two-word phrases first ('new york' -> 'ny', 'uttar pradesh' kept whole)."""
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            m = lex.get(tokens[i] + " " + tokens[i + 1])
            if m is not None:
                out.extend(m.split())
                i += 2
                continue
        out.append(tokens[i])
        i += 1
    return out


def romanize(tok: str, lex: dict) -> list:
    """Learned lexicon first (values may be phrases), then transliteration for non-Latin tokens."""
    m = lex.get(tok)
    if m is not None:
        return m.split()
    if tok.isascii():
        return [tok]
    return fold_tokens(unidecode(tok)) or [tok]


def norm_number(t: str) -> str:
    """'0050' -> '50' so zero-padded house numbers compare equal."""
    if not t.isdigit():
        return t
    return t.lstrip("0") or "0"


def ocr_fix(t: str) -> str:
    """'6allego' -> 'gallego', '5ai' -> 'sai'; leaves pure numbers and ordinals alone."""
    if t.isdigit() or t.isalpha() or _ORDINAL.match(t):
        return t
    return t.translate(_OCR)


def acronym(tokens: list) -> str:
    return "".join(t[0] for t in tokens) if len(tokens) >= 2 else ""
