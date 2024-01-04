import { atom } from 'nanostores';
import { ReactNode } from 'react';

export const $galleryHeader = atom<ReactNode | undefined>(undefined);

