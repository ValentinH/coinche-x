from __future__ import annotations

RANKS = ("7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("clubs", "diamonds", "hearts", "spades")
GLYPH_TO_RANK = {
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "10",
    "J": "J",
    "V": "J",
    "Q": "Q",
    "D": "Q",
    "K": "K",
    "R": "K",
    "A": "A",
}

RANK_TO_INDEX = {label: index for index, label in enumerate(RANKS)}
SUIT_TO_INDEX = {label: index for index, label in enumerate(SUITS)}


def canonical_rank(glyph: str) -> str:
    try:
        return GLYPH_TO_RANK[glyph]
    except KeyError as error:
        raise ValueError(f"Unknown rank glyph: {glyph}") from error


def validate_factorized_label(rank: str, suit: str, glyph: str) -> None:
    if rank not in RANK_TO_INDEX:
        raise ValueError(f"Unknown canonical rank: {rank}")
    if suit not in SUIT_TO_INDEX:
        raise ValueError(f"Unknown suit: {suit}")
    if canonical_rank(glyph) != rank:
        raise ValueError(f"Glyph {glyph} does not map to rank {rank}")
