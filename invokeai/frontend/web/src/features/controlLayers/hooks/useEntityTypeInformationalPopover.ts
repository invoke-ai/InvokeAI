import type { Feature } from 'common/components/InformationalPopover/constants';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useMemo } from 'react';

export const useEntityTypeInformationalPopover = (type: CanvasEntityIdentifier['type']): Feature | undefined => {
  const feature = useMemo(() => {
    switch (type) {
      case 'control_layer':
        return 'controlNet';
      default:
        return undefined;
    }
  }, [type]);

  return feature;
};
