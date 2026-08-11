import {
  sortCanonicalCards,
  type CanonicalCard,
  type RecognitionResult,
} from "../recognition/index.ts";
import type { CardSetDiff } from "./contracts.ts";

export function compareCardSets(
  expectedCards: readonly CanonicalCard[],
  prediction: RecognitionResult,
): CardSetDiff {
  const expected = new Set(expectedCards);
  const predicted = new Set(prediction.cards);
  const matchedCards = prediction.cards.filter((card) => expected.has(card));
  const missingCards = expectedCards.filter((card) => !predicted.has(card));
  const extraCards = prediction.cards.filter((card) => !expected.has(card));
  const { duplicateCards, invalidCardIds } = prediction.diagnostics;

  return {
    matchedCards: sortCanonicalCards(matchedCards),
    missingCards: sortCanonicalCards(missingCards),
    extraCards: sortCanonicalCards(extraCards),
    duplicateCards: [...duplicateCards],
    invalidCardIds: [...invalidCardIds],
    exact:
      missingCards.length === 0 &&
      extraCards.length === 0 &&
      duplicateCards.length === 0 &&
      invalidCardIds.length === 0,
  };
}

export function emptyFailedDiff(
  expectedCards: readonly CanonicalCard[],
): CardSetDiff {
  return {
    matchedCards: [],
    missingCards: [...expectedCards],
    extraCards: [],
    duplicateCards: [],
    invalidCardIds: [],
    exact: false,
  };
}
