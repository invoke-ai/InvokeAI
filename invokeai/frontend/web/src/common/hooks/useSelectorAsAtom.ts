import type { Selector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import type { Atom, WritableAtom } from 'nanostores';
import { atom } from 'nanostores';
import { useEffect, useState } from 'react';

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const useSelectorAsAtom = <T extends Selector<RootState, any>>(selector: T): Atom<ReturnType<T>> => {
  const store = useAppStore();
  const $atom = useState<WritableAtom<ReturnType<T>>>(() => atom<ReturnType<T>>(selector(store.getState())))[0];

  useEffect(() => {
    const unsubscribe = store.subscribe(() => {
      const prev = $atom.get();
      const next = selector(store.getState());
      if (prev !== next) {
        $atom.set(next);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [$atom, selector, store]);

  return $atom;
};
