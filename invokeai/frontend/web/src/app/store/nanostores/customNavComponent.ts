import { atom } from 'nanostores';

export const $customNavComponent = atom<undefined | (() => JSX.Element)>(undefined);
