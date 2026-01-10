import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { canvasWorkflowIntegrationOpened } from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsRunWorkflow = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();

  const onClick = useCallback(() => {
    dispatch(canvasWorkflowIntegrationOpened({ sourceEntityIdentifier: entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem onClick={onClick} icon={<PiFlowArrowBold />}>
      {t('controlLayers.workflowIntegration.runWorkflow', 'Run Workflow')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsRunWorkflow.displayName = 'CanvasEntityMenuItemsRunWorkflow';
