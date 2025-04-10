import { atom } from 'nanostores';

export const $onClickGoToModelManager = atom<(() => void) | undefined>(undefined);
