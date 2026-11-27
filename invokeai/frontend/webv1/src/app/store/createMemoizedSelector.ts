import { objectEquals } from '@observ33r/object-equals';
import { createDraftSafeSelectorCreator, createSelectorCreator, lruMemoize } from '@reduxjs/toolkit';

/**
 * A memoized selector creator that uses LRU cache and @observ33r/object-equals's objectEquals for equality check.
 */
export const createMemoizedSelector = createSelectorCreator({
  memoize: lruMemoize,
  memoizeOptions: {
    resultEqualityCheck: objectEquals,
  },
  argsMemoize: lruMemoize,
});

export const getSelectorsOptions = {
  createSelector: createDraftSafeSelectorCreator({
    memoize: lruMemoize,
    argsMemoize: lruMemoize,
  }),
};
