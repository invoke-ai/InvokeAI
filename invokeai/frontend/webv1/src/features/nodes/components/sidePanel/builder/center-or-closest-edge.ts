// Adapted from https://github.com/atlassian/pragmatic-drag-and-drop/blob/main/packages/hitbox/src/closest-edge.ts
// This adaptation adds 'center' as a possible target
import type { Input, Position } from '@atlaskit/pragmatic-drag-and-drop/types';
import type { Edge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/dist/types/types';

export type CenterOrEdge = 'center' | Edge;

// re-exporting type to make it easy to use

// When the DOM element is small, the closest-edge algorithm can result in a very small hitbox for the center
// region, making it difficult for the user to hit the center. To mitigate this, when the center is allowed,
// we use an absolute edge hitbox size of 10px or 1/4 of the element's size, whichever is smaller.

const getDistanceToCenterOrEdge: {
  [TKey in CenterOrEdge]: (
    rect: Pick<DOMRect, 'top' | 'right' | 'bottom' | 'left' | 'width' | 'height'>,
    client: Position,
    isCenterAllowed: boolean
  ) => number;
} = {
  top: (rect, client, isCenterAllowed) => {
    const distanceFromTop = Math.abs(client.y - rect.top);
    if (!isCenterAllowed) {
      return distanceFromTop;
    }
    const hitboxHeight = Math.min(rect.height / 4, 10);
    if (distanceFromTop <= hitboxHeight) {
      return 0;
    }
    return Infinity;
  },
  right: (rect, client, isCenterAllowed) => {
    const distanceFromRight = Math.abs(rect.right - client.x);
    if (!isCenterAllowed) {
      return distanceFromRight;
    }
    const hitboxWidth = Math.min(rect.width / 4, 10);
    if (distanceFromRight <= hitboxWidth) {
      return 0;
    }
    return Infinity;
  },
  bottom: (rect, client, isCenterAllowed) => {
    const distanceFromBottom = Math.abs(rect.bottom - client.y);
    if (!isCenterAllowed) {
      return distanceFromBottom;
    }
    const hitboxHeight = Math.min(rect.height / 4, 10);
    if (distanceFromBottom <= hitboxHeight) {
      return 0;
    }
    return Infinity;
  },
  left: (rect, client, isCenterAllowed) => {
    const distanceFromLeft = Math.abs(client.x - rect.left);
    if (!isCenterAllowed) {
      return distanceFromLeft;
    }
    const hitboxWidth = Math.min(rect.width / 4, 10);
    if (distanceFromLeft <= hitboxWidth) {
      return 0;
    }
    return Infinity;
  },
  center: (rect, client, _) => {
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    return Math.sqrt((client.x - centerX) ** 2 + (client.y - centerY) ** 2);
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

  const isCenterAllowed = allowedCenterOrEdge.includes('center');

  const entries = allowedCenterOrEdge.map((edge) => {
    return { edge, value: getDistanceToCenterOrEdge[edge](rect, client, isCenterAllowed) };
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
