import type { Slice, UnknownAction } from '@reduxjs/toolkit';
import type { UndoableOptions } from 'redux-undo';

export type SliceConfig<T> = {
  slice: Slice<T>;
  /**
   * A function that returns the initial state of the slice.
   */
  getInitialState: () => T;
  /**
   * The optional persist configuration for this slice. If omitted, the slice will not be persisted.
   */
  persistConfig?: {
    /**
     * Migrate the state to the current version during rehydration.
     * @param state The rehydrated state.
     * @returns A correctly-shaped state.
     */
    migrate: (state: unknown) => T;
    /**
     * Keys to omit from the persisted state.
     */
    persistDenylist?: (keyof T)[];
  };
  /**
   * The optional undoable configuration for this slice. If omitted, the slice will not be undoable.
   */
  undoableConfig?: {
    /**
     * The options to be passed into redux-undo.
     */
    reduxUndoOptions: UndoableOptions<T, UnknownAction>;
  };
};
