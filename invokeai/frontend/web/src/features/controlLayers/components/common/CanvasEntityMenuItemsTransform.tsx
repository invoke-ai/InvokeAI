import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityTransform } from 'features/controlLayers/hooks/useEntityTransform';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFrameCornersBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsTransform = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const transform = useEntityTransform(entityIdentifier);

  return (
    <MenuItem onPointerUp={transform.start} icon={<PiFrameCornersBold />} isDisabled={transform.isDisabled}>
      {t('controlLayers.transform.transform')}
    </MenuItem>
  );
});

CanvasEntityMenuItemsTransform.displayName = 'CanvasEntityMenuItemsTransform';
