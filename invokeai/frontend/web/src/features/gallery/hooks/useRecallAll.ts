import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { MetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

import { useClearStylePresetWithToast } from './useClearStylePresetWithToast';

export const useRecallAll = (imageDTO?: ImageDTO | null) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const { metadata } = useDebouncedMetadata(imageDTO?.image_name);
  const isStaging = useAppSelector(selectIsStaging);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');
  const clearStylePreset = useClearStylePresetWithToast();

  const isEnabled = useMemo(() => {
    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }

    if (tab !== 'canvas' && tab !== 'generate') {
      return false;
    }

    if (!metadata) {
      return false;
    }

    return true;
  }, [isGalleryFocused, isViewerFocused, metadata, tab]);

  const handlersToSkip = useMemo(() => {
    if (tab === 'canvas' && isStaging) {
      // When we are staging and on canvas, the bbox is locked - we cannot recall width and height
      return [MetadataHandlers.Width, MetadataHandlers.Height];
    }
    return undefined;
  }, [isStaging, tab]);

  const recall = useCallback(() => {
    if (!metadata) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    MetadataUtils.recallAll(metadata, store, handlersToSkip);
    clearStylePreset();
  }, [metadata, isEnabled, store, handlersToSkip, clearStylePreset]);

  return {
    recall,
    isEnabled,
  };
};
