import { Store } from '@reduxjs/toolkit';
import { atom } from 'nanostores';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const $store = atom<Store<any> | undefined>();
