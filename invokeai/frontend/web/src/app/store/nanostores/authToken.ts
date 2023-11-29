import { atom } from 'nanostores';

/**
 * The user's auth token.
 */
export const $authToken = atom<string | undefined>();
