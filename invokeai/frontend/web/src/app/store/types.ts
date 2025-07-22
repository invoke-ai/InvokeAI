import type { Slice } from '@reduxjs/toolkit';
import type { UndoableOptions } from 'redux-undo';

type StateFromSlice<T extends Slice> = T extends Slice<infer U> ? U : never;

export type SliceConfig<T extends Slice = Slice> = {
  slice: T;
  /**
   * A function that returns the initial state of the slice.
   */
  getInitialState: () => StateFromSlice<T>;
  /**
   * The optional persist configuration for this slice. If omitted, the slice will not be persisted.
   */
  persistConfig?: {
    /**
     * Migrate the state to the current version during rehydration.
     * @param state The rehydrated state.
     * @returns A correctly-shaped state.
     */
    migrate: (state: unknown) => StateFromSlice<T>;
    /**
     * Keys to omit from the persisted state.
     */
    persistDenylist?: (keyof StateFromSlice<T>)[];
  };
  /**
   * The optional undoable configuration for this slice. If omitted, the slice will not be undoable.
   */
  undoableConfig?: {
    /**
     * The options to be passed into redux-undo.
     */
    reduxUndoOptions: UndoableOptions<StateFromSlice<T>>;
  };
};
