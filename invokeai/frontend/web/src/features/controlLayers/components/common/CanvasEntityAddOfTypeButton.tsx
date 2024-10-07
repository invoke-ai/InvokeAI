import { IconButton } from '@invoke-ai/ui-library';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import {
  useAddControlLayer,
  useAddGlobalReferenceImage,
  useAddInpaintMask,
  useAddRasterLayer,
  useAddRegionalGuidance,
} from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  type: CanvasEntityIdentifier['type'];
};

export const CanvasEntityAddOfTypeButton = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const addInpaintMask = useAddInpaintMask();
  const addRegionalGuidance = useAddRegionalGuidance();
  const addRasterLayer = useAddRasterLayer();
  const addControlLayer = useAddControlLayer();
  const addGlobalReferenceImage = useAddGlobalReferenceImage();

  const onClick = useCallback(() => {
    switch (type) {
      case 'inpaint_mask':
        addInpaintMask();
        break;
      case 'regional_guidance':
        addRegionalGuidance();
        break;
      case 'raster_layer':
        addRasterLayer();
        break;
      case 'control_layer':
        addControlLayer();
        break;
      case 'reference_image':
        addGlobalReferenceImage();
        break;
    }
  }, [addControlLayer, addGlobalReferenceImage, addInpaintMask, addRasterLayer, addRegionalGuidance, type]);

  const label = useMemo(() => {
    switch (type) {
      case 'inpaint_mask':
        return t('controlLayers.addInpaintMask');
      case 'regional_guidance':
        return t('controlLayers.addRegionalGuidance');
      case 'raster_layer':
        return t('controlLayers.addRasterLayer');
      case 'control_layer':
        return t('controlLayers.addControlLayer');
      case 'reference_image':
        return t('controlLayers.addGlobalReferenceImage');
    }
  }, [type, t]);

  return (
    <IconButton
      size="sm"
      aria-label={label}
      tooltip={label}
      variant="link"
      icon={<NewLayerIcon />}
      onClick={onClick}
      alignSelf="stretch"
      isDisabled={isBusy}
    />
  );
});

CanvasEntityAddOfTypeButton.displayName = 'CanvasEntityAddOfTypeButton';
