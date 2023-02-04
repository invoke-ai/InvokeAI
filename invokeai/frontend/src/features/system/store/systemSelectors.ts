import { RootState } from 'app/store';

export const systemSelector = (state: RootState) => state.system;

export const toastQueueSelector = (state: RootState) => state.system.toastQueue;
