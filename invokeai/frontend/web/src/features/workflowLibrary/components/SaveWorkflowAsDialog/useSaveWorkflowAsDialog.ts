import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { getWorkflowCopyName } from 'features/workflowLibrary/util/getWorkflowCopyName';
import { atom } from 'nanostores';
import { useCallback } from 'react';

const $isOpen = atom(false);
const $workflowName = atom('');
const $shouldSaveToProject = atom(false);

export const useSaveWorkflowAsDialog = () => {
  const currentWorkflowName = useAppSelector((s) => s.workflow.name);

  const isOpen = useStore($isOpen);
  const onOpen = useCallback(() => {
    $workflowName.set(currentWorkflowName.length ? getWorkflowCopyName(currentWorkflowName) : '');
    $isOpen.set(true);
  }, [currentWorkflowName]);
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
