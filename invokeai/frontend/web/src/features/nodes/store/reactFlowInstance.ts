import { atom } from 'nanostores';
import type { ReactFlowInstance } from 'reactflow';

export const $flow = atom<ReactFlowInstance | null>(null);
export const $needsFit = atom<boolean>(true);
