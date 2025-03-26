import { atom } from 'nanostores';

export const $isDeploying = atom(false);
export const $outputNodeId = atom<string | null>(null);
