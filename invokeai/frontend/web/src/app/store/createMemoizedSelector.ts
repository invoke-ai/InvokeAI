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

export const getSelectorsOptions: GetSelectorsOptions = {
  createSelector: createDraftSafeSelectorCreator({
    memoize: lruMemoize,
    argsMemoize: lruMemoize,
  }),
};
