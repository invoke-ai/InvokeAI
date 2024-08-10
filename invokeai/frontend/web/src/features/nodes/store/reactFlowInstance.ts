import type { ReactFlowInstance } from '@xyflow/react';
import { atom } from 'nanostores';

export const $flow = atom<ReactFlowInstance | null>(null);
export const $needsFit = atom<boolean>(true);
