import type { ToastConfig } from 'features/toast/toast';
import { atom } from 'nanostores';

export const $toastMap = atom<Record<string, ToastConfig> | undefined>(undefined);
