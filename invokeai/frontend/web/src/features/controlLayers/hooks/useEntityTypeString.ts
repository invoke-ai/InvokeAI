import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const useEntityTypeString = (type: CanvasEntityIdentifier['type'], plural: boolean = false): string => {
  const { t } = useTranslation();

  const typeString = useMemo(() => {
    switch (type) {
      case 'control_layer':
        return plural ? t('controlLayers.controlLayer_withCount_other') : t('controlLayers.controlLayer');
      case 'raster_layer':
        return plural ? t('controlLayers.rasterLayer_withCount_other') : t('controlLayers.rasterLayer');
      case 'inpaint_mask':
        return plural ? t('controlLayers.inpaintMask_withCount_other') : t('controlLayers.inpaintMask');
      case 'regional_guidance':
        return plural ? t('controlLayers.regionalGuidance_withCount_other') : t('controlLayers.regionalGuidance');
      default:
        return '';
    }
  }, [type, plural, t]);

  return typeString;
};
