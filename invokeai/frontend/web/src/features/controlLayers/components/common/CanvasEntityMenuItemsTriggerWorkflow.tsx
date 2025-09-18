import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityWorkflowTrigger } from 'features/controlLayers/hooks/useEntityWorkflowTrigger';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlayCircleBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsTriggerWorkflow = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const trigger = useEntityWorkflowTrigger(entityIdentifier);

  return (
    <MenuItem onClick={trigger.start} icon={<PiPlayCircleBold />} isDisabled={trigger.isDisabled}>
      {t('controlLayers.triggerWorkflow.menuItem')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsTriggerWorkflow.displayName = 'CanvasEntityMenuItemsTriggerWorkflow';
