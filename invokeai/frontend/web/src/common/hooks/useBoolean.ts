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
export const useBoolean = (initialValue: boolean): UseBoolean => {
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

type UseDisclosure = {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  set: (isOpen: boolean) => void;
  toggle: () => void;
};

/**
 * This is the same as `buildUseBoolean`, but the method names are more descriptive,
 * serving the semantics of a disclosure state.
 *
 * Creates a hook to manage a boolean state. The boolean is stored in a nanostores atom.
 * Returns a tuple containing the hook and the atom. Use this for global boolean state.
 *
 * @param defaultIsOpen Initial state of the disclosure
 */
export const buildUseDisclosure = (defaultIsOpen: boolean): [() => UseDisclosure, WritableAtom<boolean>] => {
  const $isOpen = atom(defaultIsOpen);

  const open = () => {
    $isOpen.set(true);
  };
  const close = () => {
    $isOpen.set(false);
  };
  const set = (isOpen: boolean) => {
    $isOpen.set(isOpen);
  };
  const toggle = () => {
    $isOpen.set(!$isOpen.get());
  };

  const useDisclosure = () => {
    const isOpen = useStore($isOpen);

    return {
      isOpen,
      open,
      close,
      set,
      toggle,
    };
  };

  return [useDisclosure, $isOpen] as const;
};

/**
 * This is the same as `useBoolean`, but the method names are more descriptive,
 * serving the semantics of a disclosure state.
 *
 * Hook to manage a boolean state. Use this for a local boolean state.
 * @param defaultIsOpen Initial state of the disclosure
 *
 * @knipignore
 */
export const useDisclosure = (defaultIsOpen: boolean): UseDisclosure => {
  const [isOpen, set] = useState(defaultIsOpen);

  const open = useCallback(() => {
    set(true);
  }, [set]);
  const close = useCallback(() => {
    set(false);
  }, [set]);
  const toggle = useCallback(() => {
    set((val) => !val);
  }, [set]);

  return {
    isOpen,
    open,
    close,
    set,
    toggle,
  };
};
