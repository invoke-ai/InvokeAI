import { atom, computed } from 'nanostores';

/**
 * The user's auth token.
 */
export const $authToken = atom<string | undefined>();

/**
 * The crossOrigin value to use for all image loading. Depends on whether the user is authenticated.
 */
export const $crossOrigin = computed($authToken, (token) => (token ? 'use-credentials' : 'anonymous'));
