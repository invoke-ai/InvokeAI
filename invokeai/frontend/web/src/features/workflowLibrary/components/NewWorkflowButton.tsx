import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';

const UploadWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleNewWorkflow = useCallback(() => {
    dispatch(nodeEditorReset());
    dispatch(workflowModeChanged('edit'));

    dispatch(
      addToast(
        makeToast({
          title: t('workflows.newWorkflowCreated'),
          status: 'success',
        })
      )
    );
  }, [dispatch, t]);

  return (
    <IconButton
      aria-label={t('nodes.newWorkflow')}
      tooltip={t('nodes.newWorkflow')}
      icon={<PiFlowArrowBold />}
      onClick={handleNewWorkflow}
      pointerEvents="auto"
    />
  );
};

export default memo(UploadWorkflowMenuItem);
