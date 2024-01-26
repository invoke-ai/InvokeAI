import { atom } from 'nanostores';
import { WorkflowCategory } from '../../../features/nodes/types/workflow';

export const $workflowCategories = atom<WorkflowCategory[]>(["user", "default"]);
