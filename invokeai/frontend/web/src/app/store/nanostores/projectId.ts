import { atom } from 'nanostores';

/**
 * The optional project-id header.
 */
export const $projectId = atom<string | undefined>();

export const $projectName = atom<string | undefined>();
export const $projectUrl = atom<string | undefined>();
