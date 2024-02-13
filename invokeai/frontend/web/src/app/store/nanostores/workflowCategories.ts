import type { WorkflowCategory } from 'features/nodes/types/workflow';
import { atom } from 'nanostores';

export const $workflowCategories = atom<WorkflowCategory[]>(['user', 'default']);
