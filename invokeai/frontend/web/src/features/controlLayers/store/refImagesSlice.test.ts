import { describe, expect, it } from 'vitest';

import { refImagesReordered, refImagesSliceConfig } from './refImagesSlice';
import type { RefImagesState } from './types';
import { getReferenceImageState } from './util';

const buildState = (ids: string[]): RefImagesState => ({
  selectedEntityId: ids[0] ?? null,
  isPanelOpen: false,
  entities: ids.map((id) => getReferenceImageState(id)),
});

describe('refImagesSlice', () => {
  const { reducer } = refImagesSliceConfig.slice;

  describe('refImagesReordered', () => {
    it('reorders entities to match the provided id order', () => {
      const state = buildState(['a', 'b', 'c']);
      const result = reducer(state, refImagesReordered({ ids: ['c', 'a', 'b'] }));
      expect(result.entities.map((e) => e.id)).toEqual(['c', 'a', 'b']);
    });

    it('swaps two entities', () => {
      const state = buildState(['a', 'b']);
      const result = reducer(state, refImagesReordered({ ids: ['b', 'a'] }));
      expect(result.entities.map((e) => e.id)).toEqual(['b', 'a']);
    });

    it('reverses the list', () => {
      const state = buildState(['a', 'b', 'c', 'd']);
      const result = reducer(state, refImagesReordered({ ids: ['d', 'c', 'b', 'a'] }));
      expect(result.entities.map((e) => e.id)).toEqual(['d', 'c', 'b', 'a']);
    });

    it('preserves entity config when reordering', () => {
      const state = buildState(['a', 'b']);
      state.entities[0]!.isEnabled = false;
      const result = reducer(state, refImagesReordered({ ids: ['b', 'a'] }));
      const movedA = result.entities.find((e) => e.id === 'a');
      expect(movedA?.isEnabled).toBe(false);
    });

    it('is a no-op when the ids length does not match the entities length', () => {
      const state = buildState(['a', 'b', 'c']);
      const result = reducer(state, refImagesReordered({ ids: ['a', 'b'] }));
      expect(result.entities.map((e) => e.id)).toEqual(['a', 'b', 'c']);
    });

    it('is a no-op when ids contain an unknown id', () => {
      const state = buildState(['a', 'b', 'c']);
      const result = reducer(state, refImagesReordered({ ids: ['a', 'b', 'x'] }));
      expect(result.entities.map((e) => e.id)).toEqual(['a', 'b', 'c']);
    });

    it('is a no-op when ids contain a duplicate', () => {
      // Duplicates imply one of the original ids is missing, so length-or-map-lookup fails.
      const state = buildState(['a', 'b', 'c']);
      const result = reducer(state, refImagesReordered({ ids: ['a', 'a', 'b'] }));
      expect(result.entities.map((e) => e.id)).toEqual(['a', 'b', 'c']);
    });

    it('handles an empty list', () => {
      const state = buildState([]);
      const result = reducer(state, refImagesReordered({ ids: [] }));
      expect(result.entities).toEqual([]);
    });

    it('does not change selectedEntityId or isPanelOpen', () => {
      const state: RefImagesState = {
        ...buildState(['a', 'b', 'c']),
        selectedEntityId: 'b',
        isPanelOpen: true,
      };
      const result = reducer(state, refImagesReordered({ ids: ['c', 'b', 'a'] }));
      expect(result.selectedEntityId).toBe('b');
      expect(result.isPanelOpen).toBe(true);
    });
  });
});
