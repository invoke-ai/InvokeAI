import { RootState } from 'app/store';
import { SystemState } from './systemSlice';

export const systemSelector = (state: RootState): SystemState => state.system;

export const toastQueueSelector = (state: RootState) => state.system.toastQueue;
