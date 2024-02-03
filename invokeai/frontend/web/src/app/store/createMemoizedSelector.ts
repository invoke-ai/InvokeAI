import { createDraftSafeSelectorCreator, createSelectorCreator, lruMemoize } from '@reduxjs/toolkit';
import type { GetSelectorsOptions } from '@reduxjs/toolkit/dist/entities/state_selectors';
import { isEqual } from 'lodash-es';

/**
 * A memoized selector creator that uses LRU cache and lodash's isEqual for equality check.
 */
export const createMemoizedSelector = createSelectorCreator({
  memoize: lruMemoize,
  memoizeOptions: {
    resultEqualityCheck: isEqual,
  },
  argsMemoize: lruMemoize,
});

/**
 * A memoized selector creator that uses LRU cache default shallow equality check.
 */
export const createLruSelector = createSelectorCreator({
  memoize: lruMemoize,
  argsMemoize: lruMemoize,
});

export const createLruDraftSafeSelector = createDraftSafeSelectorCreator({
  memoize: lruMemoize,
  argsMemoize: lruMemoize,
});

export const getSelectorsOptions: GetSelectorsOptions = {
  createSelector: createLruDraftSafeSelector,
};
