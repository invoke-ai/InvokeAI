import { atom } from 'nanostores';

export const $galleryHeader = atom<undefined | (() => JSX.Element)>(undefined);
