import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const useEntityTypeString = (type: CanvasEntityIdentifier['type']): string => {
  const { t } = useTranslation();

  const typeString = useMemo(() => {
    switch (type) {
      case 'control_layer':
        return t('controlLayers.controlLayer', { count: 0 });
      case 'raster_layer':
        return t('controlLayers.rasterLayer', { count: 0 });
      case 'inpaint_mask':
        return t('controlLayers.inpaintMask', { count: 0 });
      case 'regional_guidance':
        return t('controlLayers.regionalGuidance', { count: 0 });
      case 'ip_adapter':
        return t('controlLayers.ipAdapter', { count: 0 });
      default:
        return '';
    }
  }, [type, t]);

  return typeString;
};
