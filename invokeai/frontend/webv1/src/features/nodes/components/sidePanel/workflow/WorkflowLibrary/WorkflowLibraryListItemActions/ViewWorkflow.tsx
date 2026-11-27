import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { workflowModeChanged } from 'features/nodes/store/workflowLibrarySlice';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold } from 'react-icons/pi';

export const ViewWorkflow = ({ workflowId }: { workflowId: string }) => {
  const dispatch = useAppDispatch();
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();
  const { t } = useTranslation();

  const handleClickLoad = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      loadWorkflowWithDialog({
        type: 'library',
        data: workflowId,
        onSuccess: () => {
          dispatch(workflowModeChanged('view'));
        },
      });
    },
    [dispatch, loadWorkflowWithDialog, workflowId]
  );

  return (
    <Tooltip label={t('workflows.view')} closeOnScroll>
      <IconButton
        size="sm"
        variant="link"
        alignSelf="stretch"
        aria-label={t('workflows.view')}
        onClick={handleClickLoad}
        icon={<PiEyeBold />}
      />
    </Tooltip>
  );
};
