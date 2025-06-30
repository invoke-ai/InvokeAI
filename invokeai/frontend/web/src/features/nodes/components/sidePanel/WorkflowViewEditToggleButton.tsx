import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowMode, workflowModeChanged } from 'features/nodes/store/workflowLibrarySlice';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { VIEWER_PANEL_ID, WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { MouseEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiPencilSimpleFill } from 'react-icons/pi';

export const WorkflowViewEditToggleButton = memo(() => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectWorkflowMode);
  const { t } = useTranslation();
  const { focusPanel } = useAutoLayoutContext();

  const onClickEdit = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      // Navigate to workflows tab and focus the Workflow Editor panel
      dispatch(setActiveTab('workflows'));
      dispatch(workflowModeChanged('edit'));
      // Focus the Workflow Editor panel
      focusPanel(WORKSPACE_PANEL_ID);
    },
    [dispatch, focusPanel]
  );

  const onClickView = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      // Navigate to workflows tab and focus the Image Viewer panel
      dispatch(setActiveTab('workflows'));
      dispatch(workflowModeChanged('view'));
      // Focus the Image Viewer panel
      focusPanel(VIEWER_PANEL_ID);
    },
    [dispatch, focusPanel]
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
