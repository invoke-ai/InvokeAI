import type { Slice } from '@reduxjs/toolkit';
import type { ZodType } from 'zod';

type StateFromSlice<T extends Slice> = T extends Slice<infer U> ? U : never;

export type SliceConfig<T extends Slice, TInternalState = StateFromSlice<T>, TSerializedState = StateFromSlice<T>> = {
  /**
   * The redux slice (return of createSlice).
   */
  slice: T;
  /**
   * The zod schema for the slice.
   */
  schema: ZodType<StateFromSlice<T>>;
  /**
   * A function that returns the initial state of the slice.
   */
  getInitialState: () => TSerializedState;
  /**
   * The optional persist configuration for this slice. If omitted, the slice will not be persisted.
   */
  persistConfig?: {
    /**
     * Migrate the state to the current version during rehydration. This method should throw an error if the migration
     * fails.
     *
     * @param state The rehydrated state.
     * @returns A correctly-shaped state.
     */
    migrate: (state: unknown) => TSerializedState;
    /**
     * Keys to omit from the persisted state.
     */
    persistDenylist?: (keyof StateFromSlice<T>)[];
    /**
     * Wraps state into state with history
     *
     * @param state The state without history
     * @returns The state with history
     */
    wrapState?: (state: unknown) => TInternalState;
    /**
     * Unwraps state with history
     *
     * @param state The state with history
     * @returns The state without history
     */
    unwrapState?: (state: TInternalState) => TSerializedState;
  };
};
