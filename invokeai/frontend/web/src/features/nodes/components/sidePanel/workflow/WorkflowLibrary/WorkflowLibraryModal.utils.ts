import type { WorkflowLibraryView } from 'features/nodes/store/workflowLibrarySlice';

type GetSyncedWorkflowLibraryViewArg = {
  view: WorkflowLibraryView;
  recentWorkflowsCount: number;
  yourWorkflowsCount: number;
};

export const getSyncedWorkflowLibraryView = ({
  view,
  recentWorkflowsCount,
  yourWorkflowsCount,
}: GetSyncedWorkflowLibraryViewArg): WorkflowLibraryView => {
  if (recentWorkflowsCount === 0 && view === 'recent') {
    return yourWorkflowsCount > 0 ? 'yours' : 'defaults';
  }

  if (yourWorkflowsCount === 0 && view === 'yours') {
    return recentWorkflowsCount > 0 ? 'recent' : 'defaults';
  }

  return view;
};
