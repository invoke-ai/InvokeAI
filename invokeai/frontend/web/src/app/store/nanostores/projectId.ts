import { atom } from 'nanostores';

/**
 * The optional project-id header.
 */
export const $projectId = atom<string | undefined>();
