import { useStore } from '@nanostores/react';
import { WrappedError } from 'common/util/result';
import type { Atom } from 'nanostores';
import { atom } from 'nanostores';
import { useCallback, useEffect, useMemo, useState } from 'react';

type SuccessState<T> = {
  status: 'success';
  value: T;
  error: null;
};

type ErrorState = {
  status: 'error';
  value: null;
  error: Error;
};

type PendingState = {
  status: 'pending';
  value: null;
  error: null;
};

type IdleState = {
  status: 'idle';
  value: null;
  error: null;
};

export type State<T> = IdleState | PendingState | SuccessState<T> | ErrorState;

type UseAsyncStateOptions = {
  immediate?: boolean;
};

type UseAsyncReturn<T> = {
  $state: Atom<State<T>>;
  trigger: () => Promise<void>;
  reset: () => void;
};

export const useAsyncState = <T>(execute: () => Promise<T>, options?: UseAsyncStateOptions): UseAsyncReturn<T> => {
  const $state = useState(() =>
    atom<State<T>>({
      status: 'idle',
      value: null,
      error: null,
    })
  )[0];

  const trigger = useCallback(async () => {
    $state.set({
      status: 'pending',
      value: null,
      error: null,
    });
    try {
      const value = await execute();
      $state.set({
        status: 'success',
        value,
        error: null,
      });
    } catch (error) {
      $state.set({
        status: 'error',
        value: null,
        error: WrappedError.wrap(error),
      });
    }
  }, [$state, execute]);

  const reset = useCallback(() => {
    $state.set({
      status: 'idle',
      value: null,
      error: null,
    });
  }, [$state]);

  useEffect(() => {
    if (options?.immediate) {
      trigger();
    }
  }, [options?.immediate, trigger]);

  const api = useMemo(
    () =>
      ({
        $state,
        trigger,
        reset,
      }) satisfies UseAsyncReturn<T>,
    [$state, trigger, reset]
  );

  return api;
};

type UseAsyncReturnReactive<T> = {
  state: State<T>;
  trigger: () => Promise<void>;
  reset: () => void;
};

export const useAsyncStateReactive = <T>(
  execute: () => Promise<T>,
  options?: UseAsyncStateOptions
): UseAsyncReturnReactive<T> => {
  const { $state, trigger, reset } = useAsyncState(execute, options);
  const state = useStore($state);

  return { state, trigger, reset };
};
