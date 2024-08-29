import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const useEntityTypeString = (type: CanvasEntityIdentifier['type']): string => {
  const { t } = useTranslation();

  const typeString = useMemo(() => {
    switch (type) {
      case 'control_layer':
        return t('controlLayers.controlLayer');
      case 'raster_layer':
        return t('controlLayers.rasterLayer');
      case 'inpaint_mask':
        return t('controlLayers.inpaintMask');
      case 'regional_guidance':
        return t('controlLayers.regionalGuidance');
      case 'ip_adapter':
        return t('controlLayers.ipAdapter');
      default:
        return '';
    }
  }, [type, t]);

  return typeString;
};
