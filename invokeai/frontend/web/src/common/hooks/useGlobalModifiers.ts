import { atom, map } from 'nanostores';
import { useCallback, useEffect } from 'react';

export type Modifiers = {
  shift: boolean;
  ctrl: boolean;
  meta: boolean;
  alt: boolean;
};

export const $modifiers = map<Modifiers>({
  shift: false,
  ctrl: false,
  meta: false,
  alt: false,
});

const $subscribers = atom(0);

const listener = (e: KeyboardEvent) => {
  $modifiers.setKey('shift', e.shiftKey);
  $modifiers.setKey('ctrl', e.ctrlKey);
  $modifiers.setKey('alt', e.altKey);
  $modifiers.setKey('meta', e.metaKey);
};

export const useGlobalModifiersInit = () => {
  useEffect(() => {
    $subscribers.set($subscribers.get() + 1);

    if ($subscribers.get() === 1) {
      window.addEventListener('keydown', listener);
      window.addEventListener('keyup', listener);
    }

    return () => {
      $subscribers.set(Math.max($subscribers.get() - 1, 0));
      if ($subscribers.get() > 0) {
        return;
      }
      window.removeEventListener('keydown', listener);
      window.removeEventListener('keyup', listener);
    };
  }, []);
};

export const useGlobalModifiersSetters = () => {
  const setShift = useCallback((shift: boolean) => {
    $modifiers.setKey('shift', shift);
  }, []);
  const setCtrl = useCallback((shift: boolean) => {
    $modifiers.setKey('ctrl', shift);
  }, []);
  const setAlt = useCallback((shift: boolean) => {
    $modifiers.setKey('alt', shift);
  }, []);
  const setMeta = useCallback((shift: boolean) => {
    $modifiers.setKey('meta', shift);
  }, []);
  return { setShift, setCtrl, setAlt, setMeta };
};
