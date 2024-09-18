import { atom } from 'nanostores';

/**
 * Tracks whether or not the workflow library modal is open.
 */
export const $isWorkflowLibraryModalOpen = atom<boolean>(false);
