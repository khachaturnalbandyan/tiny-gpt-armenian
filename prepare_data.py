"""
prepare_data.py — turn Armenian Wikipedia into a character-level dataset.

Idea
----
1. Stream Armenian Wikipedia articles from Hugging Face (no full dump download).
2. Keep only lines that are (almost) purely Armenian, so the model learns
   Armenian rather than a mix of Latin/Cyrillic/wiki markup.
3. Map every remaining character to an integer id (character-level tokenizer).
4. Save the ids as uint16 binary files (train.bin / val.bin) that train.py
   reads through np.memmap, plus meta.json with the vocabulary.

Complexity: one linear pass over the text — O(N) time in the number of
characters, O(N) memory for the final string (2 bytes/char in Python for
Armenian text, so 50M chars ≈ 100 MB).

Usage (on your machine — needs internet):
    pip install datasets numpy
    python prepare_data.py --max-chars 50000000
"""

import argparse          # command-line flags
import json              # meta.json (vocabulary) and stats.json
import os                # paths and folders
import re                # whitespace cleanup

import numpy as np       # fast vectorised encoding + binary output

# ---------------------------------------------------------------------------
# Which characters we allow.
# Armenian letters live in the Unicode block U+0531–U+0587.
# ---------------------------------------------------------------------------
ARMENIAN_LETTERS = (
    {chr(c) for c in range(0x0531, 0x0557)}    # capital letters Ա–Ֆ
    | {chr(c) for c in range(0x0561, 0x0588)}  # small letters ա–ֆ and the ligature և
)
ARMENIAN_PUNCT = {
    "\u055A",  # ՚ Armenian apostrophe
    "\u055B",  # ՛ emphasis mark
    "\u055C",  # ՜ exclamation mark
    "\u055D",  # ՝ comma
    "\u055E",  # ՞ question mark
    "\u055F",  # ՟ abbreviation mark
    "\u0589",  # ։ full stop
    "\u058A",  # ֊ hyphen
}
# Common punctuation, digits and whitespace that Armenian Wikipedia also uses
# (many editors type a Latin ":" instead of the Armenian full stop "։").
COMMON = set(" \n0123456789.,:;!?-()«»—\"'")

ALLOWED = ARMENIAN_LETTERS | ARMENIAN_PUNCT | COMMON   # final character whitelist


def armenian_ratio(line: str) -> float:
    """Share of letters in `line` that are Armenian (digits/punctuation ignored)."""
    letters = [ch for ch in line if ch.isalpha()]          # only alphabetic characters count
    if not letters:                                        # a line of only digits/punctuation
        return 0.0                                         # → treat as "not Armenian"
    return sum(ch in ARMENIAN_LETTERS for ch in letters) / len(letters)


def clean_article(text: str, min_ratio: float) -> str:
    """Keep the Armenian lines of one article and strip everything else."""
    kept = []                                              # cleaned lines we will keep
    for line in text.split("\n"):                          # Wikipedia text is paragraph-per-line
        line = line.strip()                                # drop leading/trailing spaces
        if len(line) < 40:                                 # headings, list stubs, captions → skip
            continue
        if armenian_ratio(line) < min_ratio:               # mostly Latin/Cyrillic → skip
            continue
        line = "".join(ch for ch in line if ch in ALLOWED) # remove characters outside the whitelist
        line = re.sub(r" {2,}", " ", line)                 # collapse runs of spaces left by removals
        kept.append(line)
    return "\n".join(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data")                       # where the .bin files go
    parser.add_argument("--max-chars", type=int, default=50_000_000)       # stop after this many characters
    parser.add_argument("--min-article-chars", type=int, default=1_000)    # skip short (often bot-made) stubs
    parser.add_argument("--min-armenian-ratio", type=float, default=0.95)  # per-line purity threshold
    parser.add_argument("--val-fraction", type=float, default=0.1)         # last 10% of text → validation
    parser.add_argument("--snapshot", default="20231101.hy")               # Hugging Face config name
    parser.add_argument("--input-txt", default=None,
                        help="skip the download and use a local UTF-8 text file instead (for testing)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)               # create output folder if missing

    # ------------------------------------------------------------------
    # 1. Collect cleaned text until we reach --max-chars.
    # ------------------------------------------------------------------
    chunks, total, articles_seen, articles_kept = [], 0, 0, 0

    if args.input_txt:                                     # local file path: used for offline tests
        with open(args.input_txt, encoding="utf-8") as f:
            articles = [f.read()]                          # treat the whole file as one "article"
    else:
        from datasets import load_dataset                  # imported here so tests don't need it
        # streaming=True reads the parquet shards lazily, so we can stop early
        articles = (row["text"] for row in
                    load_dataset("wikimedia/wikipedia", args.snapshot, split="train", streaming=True))

    for text in articles:
        articles_seen += 1
        if len(text) < args.min_article_chars:             # Armenian Wikipedia has many tiny bot stubs
            continue
        cleaned = clean_article(text, args.min_armenian_ratio)
        if not cleaned:                                    # nothing Armenian survived the filter
            continue
        chunks.append(cleaned)
        total += len(cleaned) + 2                          # +2 for the "\n\n" article separator
        articles_kept += 1
        if articles_kept % 2_000 == 0:                     # progress line every 2k kept articles
            print(f"kept {articles_kept:,} / seen {articles_seen:,} articles, {total/1e6:.1f}M chars")
        if total >= args.max_chars:                        # enough data for a ~10M-parameter model
            break

    data = "\n\n".join(chunks)                             # blank line between articles
    print(f"final text: {len(data):,} characters from {articles_kept:,} articles")

    # ------------------------------------------------------------------
    # 2. Build the character vocabulary.
    # ------------------------------------------------------------------
    chars = sorted(set(data))                              # every distinct character, sorted for stability
    stoi = {ch: i for i, ch in enumerate(chars)}           # char → id
    print(f"vocabulary size: {len(chars)}")

    # ------------------------------------------------------------------
    # 3. Encode with a lookup table instead of a Python loop.
    #    All allowed characters are in the Basic Multilingual Plane, so
    #    UTF-16-LE gives exactly one uint16 code unit per character.
    # ------------------------------------------------------------------
    lut = np.zeros(0x10000, dtype=np.uint16)               # code point → token id
    for ch, i in stoi.items():
        lut[ord(ch)] = i
    codepoints = np.frombuffer(data.encode("utf-16-le"), dtype=np.uint16)  # O(N), vectorised
    ids = lut[codepoints]                                  # O(N) gather → token ids

    # ------------------------------------------------------------------
    # 4. Split and save.
    # ------------------------------------------------------------------
    n_val = max(1, int(len(ids) * args.val_fraction))      # size of validation split (never 0: ids[:-0] would be empty)
    ids[:-n_val].tofile(os.path.join(args.out_dir, "train.bin"))
    ids[-n_val:].tofile(os.path.join(args.out_dir, "val.bin"))

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": len(chars), "itos": chars}, f, ensure_ascii=False)

    stats = {                                              # dataset stats to report
        "articles_seen": articles_seen,
        "articles_kept": articles_kept,
        "characters": len(data),
        "megabytes_utf8": round(len(data.encode("utf-8")) / 1e6, 1),
        "vocab_size": len(chars),
        "train_tokens": int(len(ids) - n_val),
        "val_tokens": n_val,
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
    # The Hugging Face streaming reader leaves background threads that can crash
    # the interpreter at shutdown (exit 134, PyGILState_Release) AFTER all files
    # are written. Everything is saved by now, so exit immediately and cleanly.
    import sys
    sys.stdout.flush()
    os._exit(0)
