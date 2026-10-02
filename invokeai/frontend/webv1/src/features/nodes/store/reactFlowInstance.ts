import type { ReactFlowInstance } from '@xyflow/react';
import type { AnyEdge, AnyNode } from 'features/nodes/types/invocation';
import { atom } from 'nanostores';

export const $flow = atom<ReactFlowInstance<AnyNode, AnyEdge> | null>(null);
export const $needsFit = atom<boolean>(false);
