import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { MetadataUtils } from 'features/metadata/parsing';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

export const useRecallDimensions = (imageDTO: ImageDTO | null | undefined) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const isStaging = useCanvasIsStaging();

  const isEnabled = useMemo(() => {
    if (!imageDTO) {
      return false;
    }

    if (tab !== 'canvas' && tab !== 'generate') {
      return false;
    }

    if (tab === 'canvas' && isStaging) {
      return false;
    }

    return true;
  }, [imageDTO, isStaging, tab]);

  const recall = useCallback(() => {
    if (!isEnabled || !imageDTO) {
      return;
    }
    MetadataUtils.recallImageDimensions(imageDTO, store);
  }, [isEnabled, imageDTO, store]);

  return {
    recall,
    isEnabled,
  };
};
