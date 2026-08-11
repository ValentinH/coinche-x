export const RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"];
export const SUITS = ["clubs", "diamonds", "hearts", "spades"];

export const CARDS = SUITS.flatMap((suit) =>
  RANKS.map((rank) => ({ rank, suit, id: `${rank}-${suit}` })),
);

export const FRENCH_FACE_GLYPH = { J: "V", Q: "D", K: "R" };
