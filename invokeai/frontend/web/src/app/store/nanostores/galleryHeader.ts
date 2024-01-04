import { atom } from 'nanostores';
import { ReactNode } from 'react';

/**
 * The optional project name.
 */
export const $galleryHeader = atom<ReactNode | undefined>(undefined);

