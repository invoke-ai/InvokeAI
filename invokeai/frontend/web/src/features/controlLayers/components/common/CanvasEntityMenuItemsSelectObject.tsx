import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntitySegmentAnything } from 'features/controlLayers/hooks/useEntitySegmentAnything';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShapesFill } from 'react-icons/pi';

export const CanvasEntityMenuItemsSelectObject = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const segmentAnything = useEntitySegmentAnything(entityIdentifier);

  return (
    <MenuItem onClick={segmentAnything.start} icon={<PiShapesFill />} isDisabled={segmentAnything.isDisabled}>
      {t('controlLayers.selectObject.selectObject')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsSelectObject.displayName = 'CanvasEntityMenuItemsSelectObject';
