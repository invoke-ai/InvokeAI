import { atom } from 'nanostores';

export const DEFAULT_QUEUE_ID = 'default';

export const $queueId = atom<string>(DEFAULT_QUEUE_ID);
