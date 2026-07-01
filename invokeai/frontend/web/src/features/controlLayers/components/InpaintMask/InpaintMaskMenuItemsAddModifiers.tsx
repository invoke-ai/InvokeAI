import { MenuItem } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useAddInpaintMaskDenoiseLimit, useAddInpaintMaskNoise } from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const InpaintMaskMenuItemsAddModifiers = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const addInpaintMaskNoise = useAddInpaintMaskNoise(entityIdentifier);
  const addInpaintMaskDenoiseLimit = useAddInpaintMaskDenoiseLimit(entityIdentifier);

  return (
    <>
      <MenuItem onClick={addInpaintMaskNoise} isDisabled={isBusy}>
        {t('controlLayers.addImageNoise')}
      </MenuItem>
      <MenuItem onClick={addInpaintMaskDenoiseLimit} isDisabled={isBusy}>
        {t('controlLayers.addDenoiseLimit')}
      </MenuItem>
    </>
  );
});

InpaintMaskMenuItemsAddModifiers.displayName = 'InpaintMaskMenuItemsAddNoise';
