import { describe, expect, it } from 'vitest';

import { getReorderedRefImageIds } from './RefImageList.helpers';

const IDS = ['a', 'b', 'c', 'd'];

describe('getReorderedRefImageIds', () => {
  describe('no-op cases', () => {
    it('returns null when sourceId is not in the list', () => {
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'missing',
        targetId: 'b',
        closestEdgeOfTarget: 'left',
      });
      expect(result).toBeNull();
    });

    it('returns null when targetId is not in the list', () => {
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'a',
        targetId: 'missing',
        closestEdgeOfTarget: 'left',
      });
      expect(result).toBeNull();
    });

    it('returns null when sourceId and targetId are the same', () => {
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'b',
        targetId: 'b',
        closestEdgeOfTarget: 'left',
      });
      expect(result).toBeNull();
    });

    it('returns null when source is already immediately to the left of target with edge=left', () => {
      // 'a' is at index 0, 'b' is at index 1. Dropping 'a' on the left edge of 'b' is a no-op.
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'a',
        targetId: 'b',
        closestEdgeOfTarget: 'left',
      });
      expect(result).toBeNull();
    });

    it('returns null when source is already immediately to the right of target with edge=right', () => {
      // 'b' is at index 1, 'a' is at index 0. Dropping 'b' on the right edge of 'a' is a no-op.
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'b',
        targetId: 'a',
        closestEdgeOfTarget: 'right',
      });
      expect(result).toBeNull();
    });
  });

  describe('forward moves (sourceIndex < targetIndex)', () => {
    it('moves source after target when edge=right', () => {
      // Move 'a' (0) to the right of 'c' (2) → ['b','c','a','d']
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'a',
        targetId: 'c',
        closestEdgeOfTarget: 'right',
      });
      expect(result).toEqual(['b', 'c', 'a', 'd']);
    });

    it('moves source before target when edge=left', () => {
      // Move 'a' (0) to the left of 'c' (2) → ['b','a','c','d']
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'a',
        targetId: 'c',
        closestEdgeOfTarget: 'left',
      });
      expect(result).toEqual(['b', 'a', 'c', 'd']);
    });
  });

  describe('backward moves (sourceIndex > targetIndex)', () => {
    it('moves source after target when edge=right', () => {
      // Move 'd' (3) to the right of 'a' (0) → ['a','d','b','c']
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'd',
        targetId: 'a',
        closestEdgeOfTarget: 'right',
      });
      expect(result).toEqual(['a', 'd', 'b', 'c']);
    });

    it('moves source before target when edge=left', () => {
      // Move 'd' (3) to the left of 'b' (1) → ['a','d','b','c']
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'd',
        targetId: 'b',
        closestEdgeOfTarget: 'left',
      });
      expect(result).toEqual(['a', 'd', 'b', 'c']);
    });
  });

  describe('null edge', () => {
    it('moves source to the target index when closestEdgeOfTarget is null (forward)', () => {
      // pragmatic-dnd's reorderWithEdge collapses null edge to indexOfTarget destination.
      // Move 'a' (0) onto 'c' (2) with no edge → ['b','c','a','d']
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'a',
        targetId: 'c',
        closestEdgeOfTarget: null,
      });
      expect(result).toEqual(['b', 'c', 'a', 'd']);
    });

    it('moves source to the target index when closestEdgeOfTarget is null (backward)', () => {
      // Move 'd' (3) onto 'b' (1) with no edge → ['a','d','b','c']
      const result = getReorderedRefImageIds({
        ids: IDS,
        sourceId: 'd',
        targetId: 'b',
        closestEdgeOfTarget: null,
      });
      expect(result).toEqual(['a', 'd', 'b', 'c']);
    });
  });

  describe('exhaustive (sourceIndex, targetIndex, edge) matrix', () => {
    // For each ordered (source, target) pair where source !== target, verify the result for
    // each possible edge. This catches any swap of 'left'/'right' or wrong axis handling.
    type Edge = 'left' | 'right' | null;
    const edges: Edge[] = ['left', 'right', null];

    const expectedFor = (sourceIdx: number, targetIdx: number, edge: Edge): string[] | null => {
      // Hand-rolled reference implementation — independent of the helper under test.
      // Returns null for no-ops that mirror the helper's short-circuits.
      if (sourceIdx === targetIdx) {
        return null;
      }
      let edgeIndexDelta = 0;
      if (edge === 'right') {
        edgeIndexDelta = 1;
      } else if (edge === 'left') {
        edgeIndexDelta = -1;
      }
      if (sourceIdx === targetIdx + edgeIndexDelta) {
        return null;
      }
      // Compute destination per pragmatic-dnd's `getReorderDestinationIndex` for axis='horizontal'.
      let destination: number;
      if (edge === null) {
        destination = targetIdx;
      } else {
        const isGoingAfter = edge === 'right';
        const isMovingForward = sourceIdx < targetIdx;
        if (isMovingForward) {
          destination = isGoingAfter ? targetIdx : targetIdx - 1;
        } else {
          destination = isGoingAfter ? targetIdx + 1 : targetIdx;
        }
      }
      const next = IDS.slice();
      const [moved] = next.splice(sourceIdx, 1);
      next.splice(destination, 0, moved!);
      return next;
    };

    for (let s = 0; s < IDS.length; s++) {
      for (let t = 0; t < IDS.length; t++) {
        for (const edge of edges) {
          const label = `source=${s} target=${t} edge=${edge ?? 'null'}`;
          it(label, () => {
            const result = getReorderedRefImageIds({
              ids: IDS,
              sourceId: IDS[s]!,
              targetId: IDS[t]!,
              closestEdgeOfTarget: edge,
            });
            expect(result).toEqual(expectedFor(s, t, edge));
          });
        }
      }
    }
  });
});
