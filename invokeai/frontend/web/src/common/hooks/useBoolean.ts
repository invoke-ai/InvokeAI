import { useStore } from '@nanostores/react';
import type { WritableAtom } from 'nanostores';
import { atom } from 'nanostores';
import { useCallback, useState } from 'react';

type UseBoolean = {
  isTrue: boolean;
  setTrue: () => void;
  setFalse: () => void;
  set: (value: boolean) => void;
  toggle: () => void;
};

/**
 * Creates a hook to manage a boolean state. The boolean is stored in a nanostores atom.
 * Returns a tuple containing the hook and the atom. Use this for global boolean state.
 * @param initialValue Initial value of the boolean
 */
export const buildUseBoolean = (initialValue: boolean): [() => UseBoolean, WritableAtom<boolean>] => {
  const $boolean = atom(initialValue);

  const setTrue = () => {
    $boolean.set(true);
  };
  const setFalse = () => {
    $boolean.set(false);
  };
  const set = (value: boolean) => {
    $boolean.set(value);
  };
  const toggle = () => {
    $boolean.set(!$boolean.get());
  };

  const useBoolean = () => {
    const isTrue = useStore($boolean);

    return {
      isTrue,
      setTrue,
      setFalse,
      set,
      toggle,
    };
  };

  return [useBoolean, $boolean] as const;
};

/**
 * Hook to manage a boolean state. Use this for a local boolean state.
 * @param initialValue Initial value of the boolean
 */
export const useBoolean = (initialValue: boolean) => {
  const [isTrue, set] = useState(initialValue);

  const setTrue = useCallback(() => {
    set(true);
  }, [set]);
  const setFalse = useCallback(() => {
    set(false);
  }, [set]);
  const toggle = useCallback(() => {
    set((val) => !val);
  }, [set]);

  return {
    isTrue,
    setTrue,
    setFalse,
    set,
    toggle,
  };
};
