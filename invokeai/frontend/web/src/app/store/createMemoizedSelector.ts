import {
  createDraftSafeSelectorCreator,
  createSelectorCreator,
  weakMapMemoize,
} from '@reduxjs/toolkit';
import type { GetSelectorsOptions } from '@reduxjs/toolkit/dist/entities/state_selectors';

/**
 * A memoized selector creator that uses LRU cache and lodash's isEqual for equality check.
 */
export const createMemoizedSelector = createSelectorCreator({
  memoize: weakMapMemoize,
  // memoizeOptions: {
  //   resultEqualityCheck: isEqual,
  // },
  argsMemoize: weakMapMemoize,
});

/**
 * A memoized selector creator that uses LRU cache default shallow equality check.
 */
export const createLruSelector = createSelectorCreator({
  memoize: weakMapMemoize,
  argsMemoize: weakMapMemoize,
});

export const createLruDraftSafeSelector = createDraftSafeSelectorCreator({
  memoize: weakMapMemoize,
  argsMemoize: weakMapMemoize,
});

export const getSelectorsOptions: GetSelectorsOptions = {
  createSelector: createLruDraftSafeSelector,
};
