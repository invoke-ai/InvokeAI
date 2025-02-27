import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityTransform } from 'features/controlLayers/hooks/useEntityTransform';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold, PiFrameCornersBold } from 'react-icons/pi';

export const CanvasEntityMenuItemsTransform = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const transform = useEntityTransform(entityIdentifier);

  return (
    <>
      <MenuItem onClick={transform.start} icon={<PiFrameCornersBold />} isDisabled={transform.isDisabled}>
        {t('controlLayers.transform.transform')}
      </MenuItem>
      <MenuItem onClick={transform.fitToBbox} icon={<PiBoundingBoxBold />} isDisabled={transform.isDisabled}>
        {t('controlLayers.transform.fitToBbox')}
      </MenuItem>
    </>
  );
});

CanvasEntityMenuItemsTransform.displayName = 'CanvasEntityMenuItemsTransform';
