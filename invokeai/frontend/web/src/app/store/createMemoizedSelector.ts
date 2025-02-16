import { createDraftSafeSelectorCreator, createSelectorCreator, lruMemoize } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
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

export const getSelectorsOptions = {
  createSelector: createDraftSafeSelectorCreator({
    memoize: lruMemoize,
    argsMemoize: lruMemoize,
  }),
};

export const createMemoizedAppSelector = createMemoizedSelector.withTypes<RootState>();
