// Adapted from https://github.com/atlassian/pragmatic-drag-and-drop/blob/main/packages/hitbox/src/closest-edge.ts
// This adaptation adds 'center' as a possible target
import type { Input, Position } from '@atlaskit/pragmatic-drag-and-drop/types';
import type { Edge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/dist/types/types';

export type CenterOrEdge = 'center' | Edge;

const CENTER_BIAS_FACTOR = 0.8;

// re-exporting type to make it easy to use

const getDistanceToCenterOrEdge: {
  [TKey in CenterOrEdge]: (rect: DOMRect, client: Position) => number;
} = {
  top: (rect, client) => Math.abs(client.y - rect.top),
  right: (rect, client) => Math.abs(rect.right - client.x),
  bottom: (rect, client) => Math.abs(rect.bottom - client.y),
  left: (rect, client) => Math.abs(client.x - rect.left),
  center: (rect, client) => {
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    return Math.sqrt((client.x - centerX) ** 2 + (client.y - centerY) ** 2) * CENTER_BIAS_FACTOR;
  },
};

// using a symbol so we can guarantee a key with a unique value
const uniqueKey = Symbol('centerWithClosestEdge');

/**
 * Adds a unique `Symbol` to the `userData` object. Use with `extractClosestEdge()` for type safe lookups.
 */
export function attachClosestCenterOrEdge(
  userData: Record<string | symbol, unknown>,
  {
    element,
    input,
    allowedCenterOrEdge,
  }: {
    element: Element;
    input: Input;
    allowedCenterOrEdge: CenterOrEdge[];
  }
): Record<string | symbol, unknown> {
  const client: Position = {
    x: input.clientX,
    y: input.clientY,
  };
  // I tried caching the result of `getBoundingClientRect()` for a single
  // frame in order to improve performance.
  // However, on measurement I saw no improvement. So no longer caching
  const rect: DOMRect = element.getBoundingClientRect();
  const entries = allowedCenterOrEdge.map((edge) => {
    return {
      edge,
      value: getDistanceToCenterOrEdge[edge](rect, client),
    };
  });

  // edge can be `null` when `allowedCenterOrEdge` is []
  const addClosestCenterOrEdge: CenterOrEdge | null = entries.sort((a, b) => a.value - b.value)[0]?.edge ?? null;

  return {
    ...userData,
    [uniqueKey]: addClosestCenterOrEdge,
  };
}

/**
 * Returns the value added by `attachClosestEdge()` to the `userData` object. It will return `null` if there is no value.
 */
export function extractClosestCenterOrEdge(userData: Record<string | symbol, unknown>): CenterOrEdge | null {
  return (userData[uniqueKey] as CenterOrEdge) ?? null;
}
