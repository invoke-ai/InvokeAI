import { atom } from 'nanostores';
import type { ReactNode } from 'react';

export const $headerComponent = atom<ReactNode | undefined>(undefined);
