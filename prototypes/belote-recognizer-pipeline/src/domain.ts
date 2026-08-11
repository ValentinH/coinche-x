export const RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"] as const;
export const SUITS = ["clubs", "diamonds", "hearts", "spades"] as const;
export const RANK_GLYPHS = [
  "7",
  "8",
  "9",
  "10",
  "J",
  "Q",
  "K",
  "V",
  "D",
  "R",
  "A",
] as const;

export type Rank = (typeof RANKS)[number];
export type Suit = (typeof SUITS)[number];
export type RankGlyph = (typeof RANK_GLYPHS)[number];

export interface Card {
  rank: Rank;
  suit: Suit;
}

export const CARDS: readonly Card[] = SUITS.flatMap((suit) =>
  RANKS.map((rank) => ({ rank, suit })),
);

const GLYPH_TO_RANK: Readonly<Record<RankGlyph, Rank>> = {
  "7": "7",
  "8": "8",
  "9": "9",
  "10": "10",
  J: "J",
  V: "J",
  Q: "Q",
  D: "Q",
  K: "K",
  R: "K",
  A: "A",
};

export function canonicalRankForGlyph(glyph: RankGlyph): Rank {
  return GLYPH_TO_RANK[glyph];
}

export function cardId(card: Card): string {
  return `${card.rank}-${card.suit}`;
}

export function isRank(value: unknown): value is Rank {
  return typeof value === "string" && RANKS.includes(value as Rank);
}

export function isSuit(value: unknown): value is Suit {
  return typeof value === "string" && SUITS.includes(value as Suit);
}
