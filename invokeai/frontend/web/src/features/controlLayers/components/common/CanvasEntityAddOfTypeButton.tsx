import { IconButton } from '@invoke-ai/ui-library';
import {
  useAddControlLayer,
  useAddInpaintMask,
  useAddIPAdapter,
  useAddRasterLayer,
  useAddRegionalGuidance,
} from 'features/controlLayers/hooks/addLayerHooks';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

type Props = {
  type: CanvasEntityIdentifier['type'];
};

export const CanvasEntityAddOfTypeButton = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const addInpaintMask = useAddInpaintMask();
  const addRegionalGuidance = useAddRegionalGuidance();
  const addRasterLayer = useAddRasterLayer();
  const addControlLayer = useAddControlLayer();
  const addIPAdapter = useAddIPAdapter();

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
      case 'ip_adapter':
        addIPAdapter();
        break;
    }
  }, [addControlLayer, addIPAdapter, addInpaintMask, addRasterLayer, addRegionalGuidance, type]);

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
