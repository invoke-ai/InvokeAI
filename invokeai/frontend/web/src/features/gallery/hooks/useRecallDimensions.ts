import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { MetadataUtils } from 'features/metadata/parsing';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

export const useRecallDimensions = (imageDTO?: ImageDTO | null) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const isStaging = useAppSelector(selectIsStaging);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');

  const isEnabled = useMemo(() => {
    if (!imageDTO) {
      return false;
    }

    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }

    if (tab !== 'canvas' && tab !== 'generate') {
      return false;
    }

    if (tab === 'canvas' && isStaging) {
      return false;
    }

    return true;
  }, [imageDTO, isGalleryFocused, isStaging, isViewerFocused, tab]);

  const recall = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    MetadataUtils.recallDimensions(imageDTO, store);
  }, [isEnabled, imageDTO, store]);

  return {
    recall,
    isEnabled,
  };
};
