import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  canvasWorkflowIntegrationOpened,
  type CanvasWorkflowSourceEntityIdentifier,
} from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsRunWorkflow = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();

  if (entityIdentifier.type === 'vector_layer') {
    return null;
  }

  const sourceEntityIdentifier: CanvasWorkflowSourceEntityIdentifier = {
    id: entityIdentifier.id,
    type: entityIdentifier.type,
  };

  const onClick = useCallback(() => {
    dispatch(canvasWorkflowIntegrationOpened({ sourceEntityIdentifier }));
  }, [dispatch, sourceEntityIdentifier]);

  return (
    <MenuItem onClick={onClick} icon={<PiFlowArrowBold />}>
      {t('controlLayers.workflowIntegration.runWorkflow')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsRunWorkflow.displayName = 'CanvasEntityMenuItemsRunWorkflow';
