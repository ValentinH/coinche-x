export const CARD_RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"] as const;
export const CARD_SUITS = ["C", "D", "H", "S"] as const;

export type CardRank = (typeof CARD_RANKS)[number];
export type CardSuit = (typeof CARD_SUITS)[number];
export type CanonicalCard = `${CardRank}${CardSuit}`;

const CARD_ORDER = new Map<CanonicalCard, number>(
  CARD_RANKS.flatMap((rank) =>
    CARD_SUITS.map((suit) => `${rank}${suit}` as CanonicalCard),
  ).map((card, index) => [card, index]),
);

export function isCanonicalCard(value: unknown): value is CanonicalCard {
  return typeof value === "string" && CARD_ORDER.has(value as CanonicalCard);
}

export function sortCanonicalCards(
  cards: Iterable<CanonicalCard>,
): CanonicalCard[] {
  return [...cards].sort(
    (left, right) => CARD_ORDER.get(left)! - CARD_ORDER.get(right)!,
  );
}

export function canonicalCardSet(values: readonly unknown[]): {
  cards: CanonicalCard[];
  duplicateCards: CanonicalCard[];
  invalidCardIds: string[];
} {
  const cards = new Set<CanonicalCard>();
  const duplicateCards: CanonicalCard[] = [];
  const invalidCardIds: string[] = [];

  for (const value of values) {
    if (!isCanonicalCard(value)) {
      invalidCardIds.push(renderInvalidValue(value));
      continue;
    }

    if (cards.has(value)) {
      duplicateCards.push(value);
      continue;
    }

    cards.add(value);
  }

  return {
    cards: sortCanonicalCards(cards),
    duplicateCards: sortCanonicalCards(duplicateCards),
    invalidCardIds,
  };
}

function renderInvalidValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}
