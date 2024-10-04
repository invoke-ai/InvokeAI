import { atom } from 'nanostores';
import type { ReactNode } from 'react';

export const $customNavComponent = atom<ReactNode | undefined>(undefined);
