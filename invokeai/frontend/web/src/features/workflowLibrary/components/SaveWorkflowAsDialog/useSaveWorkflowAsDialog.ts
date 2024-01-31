import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { getWorkflowCopyName } from 'features/workflowLibrary/util/getWorkflowCopyName';
import { atom } from 'nanostores';
import { useCallback } from 'react';

const $isOpen = atom(false);
const $workflowName = atom('');
const $shouldSaveToProject = atom(false);

const selectNewWorkflowName = createSelector(selectWorkflowSlice, ({ name, id }): string => {
  // If the workflow has no ID, it's a new workflow that has never been saved to the server. The dialog should use
  // whatever the user has entered in the workflow name field.
  if (!id) {
    return name;
  }
  // Else, the workflow is already saved to the server. The dialog should use the workflow's name with " (copy)"
  // appended to it.
  if (name.length) {
    return getWorkflowCopyName(name);
  }
  // Else, we have a workflow that has been saved to the server, but has no name. This should never happen, but if
  // it does, we just return an empty string and let the dialog use the default name.
  return '';
});

export const useSaveWorkflowAsDialog = () => {
  const newWorkflowName = useAppSelector(selectNewWorkflowName);

  const isOpen = useStore($isOpen);
  const onOpen = useCallback(() => {
    $workflowName.set(newWorkflowName);
    $isOpen.set(true);
  }, [newWorkflowName]);
  const onClose = useCallback(() => {
    $isOpen.set(false);
    $workflowName.set('');
    $shouldSaveToProject.set(false);
  }, []);

  const workflowName = useStore($workflowName);
  const setWorkflowName = useCallback((workflowName: string) => $workflowName.set(workflowName), []);

  const shouldSaveToProject = useStore($shouldSaveToProject);
  const setShouldSaveToProject = useCallback((shouldSaveToProject: boolean) => {
    $shouldSaveToProject.set(shouldSaveToProject);
  }, []);

  return { workflowName, setWorkflowName, shouldSaveToProject, setShouldSaveToProject, isOpen, onOpen, onClose };
};
