import { atom } from 'nanostores';
import { ReactNode } from 'react';

export const $headerComponent = atom<ReactNode | undefined>(undefined);
