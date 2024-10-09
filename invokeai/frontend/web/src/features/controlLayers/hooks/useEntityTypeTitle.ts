import { useEntityTypeCount } from 'features/controlLayers/hooks/useEntityTypeCount';
import { useEntityTypeIsHidden } from 'features/controlLayers/hooks/useEntityTypeIsHidden';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const useEntityTypeTitle = (type: CanvasEntityIdentifier['type']): string => {
  const { t } = useTranslation();

  const isHidden = useEntityTypeIsHidden(type);
  const count = useEntityTypeCount(type);

  const title = useMemo(() => {
    const context = isHidden ? 'hidden' : 'visible';
    switch (type) {
      case 'control_layer':
        return t('controlLayers.controlLayers_withCount', { count, context });
      case 'raster_layer':
        return t('controlLayers.rasterLayers_withCount', { count, context });
      case 'inpaint_mask':
        return t('controlLayers.inpaintMasks_withCount', { count, context });
      case 'regional_guidance':
        return t('controlLayers.regionalGuidance_withCount', { count, context });
      case 'reference_image':
        return t('controlLayers.globalReferenceImages_withCount', { count, context });
      default:
        return '';
    }
  }, [type, t, count, isHidden]);

  return title;
};
