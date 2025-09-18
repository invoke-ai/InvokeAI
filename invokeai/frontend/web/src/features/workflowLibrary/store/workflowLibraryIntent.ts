import { atom } from 'nanostores';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

type WorkflowLibraryIntent =
  | { mode: 'browse' }
  | { mode: 'trigger-workflow'; onSelect: (workflow: WorkflowRecordListItemWithThumbnailDTO) => void };

export const $workflowLibraryIntent = atom<WorkflowLibraryIntent>({ mode: 'browse' });

export const setWorkflowLibraryBrowseIntent = () => {
  $workflowLibraryIntent.set({ mode: 'browse' });
};

export const setWorkflowLibraryTriggerIntent = (
  onSelect: (workflow: WorkflowRecordListItemWithThumbnailDTO) => void
) => {
  $workflowLibraryIntent.set({ mode: 'trigger-workflow', onSelect });
};
