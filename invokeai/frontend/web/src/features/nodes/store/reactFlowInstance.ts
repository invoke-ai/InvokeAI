import { atom } from 'nanostores';
import { ReactFlowInstance } from 'reactflow';

export const $flow = atom<ReactFlowInstance | null>(null);
