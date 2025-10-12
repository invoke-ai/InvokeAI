import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useActiveCanvasIsStaging } from 'features/controlLayers/hooks/useCanvasIsStaging';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import { MetadataUtils } from 'features/metadata/parsing';
import { useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

export const useRecallDimensions = (imageDTO: ImageDTO) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const isStaging = useActiveCanvasIsStaging();

  const isEnabled = useMemo(() => {
    if (tab !== 'canvas' && tab !== 'generate') {
      return false;
    }

    if (tab === 'canvas' && isStaging) {
      return false;
    }

    return true;
  }, [isStaging, tab]);

  const recall = useCallback(() => {
    if (!isEnabled) {
      return;
    }
    MetadataUtils.recallImageDimensions(imageDTO, store);
  }, [isEnabled, imageDTO, store]);

  return {
    recall,
    isEnabled,
  };
};
