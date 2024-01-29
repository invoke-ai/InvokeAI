import { atom } from 'nanostores';

export const $logo = atom<undefined | (() => JSX.Element)>(undefined);
