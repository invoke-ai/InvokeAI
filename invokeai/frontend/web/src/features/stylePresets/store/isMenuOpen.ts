import { atom } from 'nanostores';

/**
 * Tracks whether or not the style preset menu is open.
 */
export const $isMenuOpen = atom<boolean>(false);
