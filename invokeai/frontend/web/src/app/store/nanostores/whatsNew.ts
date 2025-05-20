import { atom } from 'nanostores';
import type { ReactNode } from 'react';

export const $whatsNew = atom<ReactNode[] | undefined>(undefined);
