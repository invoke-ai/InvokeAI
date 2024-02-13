import type { createStore } from 'app/store/store';
import { atom } from 'nanostores';

// Inject socket options and url into window for debugging
declare global {
  interface Window {
    $store?: typeof $store;
  }
}

export const $store = atom<Readonly<ReturnType<typeof createStore>> | undefined>();
