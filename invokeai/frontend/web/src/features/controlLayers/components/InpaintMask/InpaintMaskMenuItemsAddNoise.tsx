import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useAddInpaintMaskNoise } from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const InpaintMaskMenuItemsAddNoise = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const addInpaintMaskNoise = useAddInpaintMaskNoise(entityIdentifier);

  return (
    <MenuItem onClick={addInpaintMaskNoise} isDisabled={isBusy}>
      {t('controlLayers.addImageNoise')}
    </MenuItem>
  );
});

InpaintMaskMenuItemsAddNoise.displayName = 'InpaintMaskMenuItemsAddNoise';
