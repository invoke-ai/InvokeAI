import { atom } from 'nanostores';
import { useCallback, useEffect } from 'react';

export const $shift = atom(false);
export const $ctrl = atom(false);
export const $meta = atom(false);
export const $alt = atom(false);

const $subscribers = atom(0);

const listener = (e: KeyboardEvent) => {
  $shift.set(e.shiftKey);
  $ctrl.set(e.ctrlKey);
  $alt.set(e.altKey);
  $meta.set(e.metaKey);
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
    $shift.set(shift);
  }, []);
  const setCtrl = useCallback((ctrl: boolean) => {
    $ctrl.set(ctrl);
  }, []);
  const setAlt = useCallback((alt: boolean) => {
    $alt.set(alt);
  }, []);
  const setMeta = useCallback((meta: boolean) => {
    $meta.set(meta);
  }, []);
  return { setShift, setCtrl, setAlt, setMeta };
};
