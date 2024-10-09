import { atom } from 'nanostores';

/**
 * Tracks the state for the workflow list menu.
 */
export const $isWorkflowListMenuIsOpen = atom<boolean>(false);
