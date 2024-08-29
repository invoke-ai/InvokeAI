import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import {
  controlLayerAdded,
  inpaintMaskAdded,
  ipaAdded,
  rasterLayerAdded,
  rgAdded,
} from 'features/controlLayers/store/canvasSlice';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

type Props = {
  type: CanvasEntityIdentifier['type'];
};

export const CanvasEntityAddOfTypeButton = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    switch (type) {
      case 'inpaint_mask':
        dispatch(inpaintMaskAdded({ isSelected: true }));
        break;
      case 'regional_guidance':
        dispatch(rgAdded({ isSelected: true }));
        break;
      case 'raster_layer':
        dispatch(rasterLayerAdded({ isSelected: true }));
        break;
      case 'control_layer':
        dispatch(controlLayerAdded({ isSelected: true }));
        break;
      case 'ip_adapter':
        dispatch(ipaAdded({ isSelected: true }));
        break;
    }
  }, [dispatch, type]);

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
      case 'ip_adapter':
        return t('controlLayers.addIPAdapter');
    }
  }, [type, t]);

  return (
    <IconButton
      size="sm"
      aria-label={label}
      tooltip={label}
      variant="link"
      icon={<PiPlusBold />}
      onClick={onClick}
      alignSelf="stretch"
    />
  );
});

CanvasEntityAddOfTypeButton.displayName = 'CanvasEntityAddOfTypeButton';
