import { atom } from 'nanostores';
import type { ReactNode } from 'react';

export const $logo = atom<ReactNode | undefined>(undefined);
