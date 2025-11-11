import type { Slice } from '@reduxjs/toolkit';
import type { ZodType } from 'zod';
import { z } from 'zod';

type StateFromSlice<T extends Slice> = T extends Slice<infer U> ? U : never;
export type SerializedStateFromDenyList<S, T extends readonly (keyof S)[]> = Omit<S, T[number]>;

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
    migrate: (state: unknown) => TInternalState;
    /**
     * Serializes the state
     *
     * @param state The internal state
     * @returns The serialized state
     */
    serialize?: (state: TInternalState) => TSerializedState;
    /**
     * Deserializes the state
     *
     * @param state The serialized state
     * @returns The internal state
     */
    deserialize?: (state: unknown) => TInternalState;
  };
};

export const zStateWithHistory = <T extends z.ZodTypeAny>(stateSchema: T) =>
  z.object({
    past: z.array(stateSchema),
    present: stateSchema,
    future: z.array(stateSchema),
    _latestUnfiltered: stateSchema.optional(),
    group: z.unknown().optional(),
    index: z.number().optional(),
    limit: z.number().optional(),
  });
