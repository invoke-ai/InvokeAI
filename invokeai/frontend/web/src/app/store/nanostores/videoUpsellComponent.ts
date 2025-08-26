import { atom } from 'nanostores';
import type { ReactNode } from 'react';

export const $videoUpsellComponent = atom<ReactNode | undefined>(undefined);
