import type { Feature } from 'common/components/InformationalPopover/constants';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityTypeInformationalPopover = (type: CanvasEntityIdentifier['type']): Feature | undefined => {
  const feature = useMemo(() => {
    switch (type) {
      case 'control_layer':
        return 'controlNet';
      case 'inpaint_mask':
        return 'inpainting';
      case 'raster_layer':
        return 'rasterLayer';
      case 'regional_guidance':
        return 'regionalGuidanceAndReferenceImage';
      case 'reference_image':
        return 'globalReferenceImage';

      default:
        return undefined;
    }
  }, [type]);

  return feature;
};
