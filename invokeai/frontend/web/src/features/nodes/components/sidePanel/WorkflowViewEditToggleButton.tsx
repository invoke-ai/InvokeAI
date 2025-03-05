import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowMode, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import type { MouseEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiPencilSimpleFill } from 'react-icons/pi';

export const WorkflowViewEditToggleButton = memo(() => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectWorkflowMode);
  const { t } = useTranslation();

  const onClickEdit = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(workflowModeChanged('edit'));
    },
    [dispatch]
  );

  const onClickView = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(workflowModeChanged('view'));
    },
    [dispatch]
  );

  if (mode === 'view') {
    return (
      <IconButton
        aria-label={t('nodes.editMode')}
        tooltip={t('nodes.editMode')}
        onClick={onClickEdit}
        icon={<PiPencilSimpleFill />}
        variant="ghost"
        size="sm"
      />
    );
  }

  // mode === 'edit'
  return (
    <IconButton
      aria-label={t('nodes.viewMode')}
      tooltip={t('nodes.viewMode')}
      onClick={onClickView}
      icon={<PiEyeBold />}
      variant="ghost"
      size="sm"
    />
  );
});

WorkflowViewEditToggleButton.displayName = 'WorkflowViewEditToggleButton';
