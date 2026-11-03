import { useCallback, useSyncExternalStore } from 'react';

type ModifierKey = 'Alt' | 'Control' | 'Meta' | 'Shift';

interface ModifierStore {
  held: boolean;
  listeners: Set<() => void>;
  stop: (() => void) | null;
}

const stores = new Map<ModifierKey, ModifierStore>();

const getStore = (key: ModifierKey): ModifierStore => {
  let store = stores.get(key);

  if (!store) {
    store = { held: false, listeners: new Set(), stop: null };
    stores.set(key, store);
  }

  return store;
};

const setHeld = (store: ModifierStore, held: boolean) => {
  if (store.held !== held) {
    store.held = held;
    store.listeners.forEach((listener) => listener());
  }
};

/**
 * Window listeners for one modifier, shared by every subscriber. Key events run in the capture phase so a handler that
 * stops propagation below cannot hide a release. A release the page never hears at all (the keys went to another
 * window, which does not always blur this one) is corrected by the next pointer event, which carries the live state.
 */
const listen = (key: ModifierKey, store: ModifierStore): (() => void) => {
  const onKey = (event: KeyboardEvent) =>
    setHeld(store, event.key === key ? event.type === 'keydown' : event.getModifierState(key));
  const onPointer = (event: PointerEvent) => setHeld(store, event.getModifierState(key));
  const onBlur = () => setHeld(store, false);
  const capture = { capture: true, passive: true } as const;

  window.addEventListener('keydown', onKey, true);
  window.addEventListener('keyup', onKey, true);
  window.addEventListener('pointermove', onPointer, capture);
  window.addEventListener('pointerdown', onPointer, capture);
  window.addEventListener('blur', onBlur);

  return () => {
    window.removeEventListener('keydown', onKey, true);
    window.removeEventListener('keyup', onKey, true);
    window.removeEventListener('pointermove', onPointer, capture);
    window.removeEventListener('pointerdown', onPointer, capture);
    window.removeEventListener('blur', onBlur);
  };
};

const subscribe = (key: ModifierKey, listener: () => void): (() => void) => {
  const store = getStore(key);

  store.listeners.add(listener);
  store.stop ??= listen(key, store);

  return () => {
    store.listeners.delete(listener);

    if (store.listeners.size === 0) {
      store.stop?.();
      store.stop = null;
      // Nobody hears the key while unsubscribed, so a later subscriber must not inherit a stale press.
      store.held = false;
    }
  };
};

/** Share event-driven modifier state across widgets. */
export const useModifierHeld = (key: ModifierKey): boolean => {
  const subscribeKey = useCallback((listener: () => void) => subscribe(key, listener), [key]);
  const getSnapshot = useCallback(() => getStore(key).held, [key]);

  return useSyncExternalStore(subscribeKey, getSnapshot, getSnapshot);
};
