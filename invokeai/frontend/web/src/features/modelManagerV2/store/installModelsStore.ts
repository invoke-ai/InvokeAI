import { atom } from 'nanostores';

/**
 * Atom to manage the active tab index for the Install Models panel.
 * Moved to separate file to avoid circular dependencies.
 */
export const $installModelsTab = atom(0);
