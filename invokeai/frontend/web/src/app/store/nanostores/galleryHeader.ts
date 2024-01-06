import { atom } from 'nanostores';
import type { ReactNode } from 'react';

export const $galleryHeader = atom<ReactNode | undefined>(undefined);
