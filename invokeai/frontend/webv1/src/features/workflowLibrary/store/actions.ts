import { createAction } from '@reduxjs/toolkit';
import type { WorkflowCategory } from 'features/nodes/types/workflow';

export const workflowDownloaded = createAction('workflowLibrary/workflowDownloaded');

export const workflowLoadedFromFile = createAction('workflowLibrary/workflowLoadedFromFile');

export const newWorkflowSaved = createAction<{ category: WorkflowCategory }>('workflowLibrary/newWorkflowSaved');

export const workflowUpdated = createAction('workflowLibrary/workflowUpdated');
