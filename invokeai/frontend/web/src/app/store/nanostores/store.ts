import type { createStore } from 'app/store/store';
import { atom } from 'nanostores';

export const $store = atom<
  Readonly<ReturnType<typeof createStore>> | undefined
>();
