import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useActiveCanvasIsStaging } from 'features/controlLayers/hooks/useCanvasIsStaging';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import { ImageMetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { useCallback, useMemo } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

import { useClearStylePresetWithToast } from './useClearStylePresetWithToast';

export const useRecallRemix = (imageDTO: ImageDTO) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const isStaging = useActiveCanvasIsStaging();
  const clearStylePreset = useClearStylePresetWithToast();

  const { metadata, isLoading } = useDebouncedMetadata(imageDTO.image_name);

  const isEnabled = useMemo(() => {
    if (isLoading) {
      return false;
    }

    if (tab !== 'canvas' && tab !== 'generate') {
      return false;
    }

    if (!metadata) {
      return false;
    }

    return true;
  }, [isLoading, metadata, tab]);

  const handlersToSkip = useMemo(() => {
    // Remix always skips the seed handler
    const _handlersToSkip = [ImageMetadataHandlers.Seed];
    if (tab === 'canvas' && isStaging) {
      // When we are staging and on canvas, the bbox is locked - we cannot recall width and height
      _handlersToSkip.push(ImageMetadataHandlers.Width, ImageMetadataHandlers.Height);
    }
    return _handlersToSkip;
  }, [isStaging, tab]);

  const recall = useCallback(() => {
    if (!metadata) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    MetadataUtils.recallAllImageMetadata(metadata, store, handlersToSkip);
    clearStylePreset();
  }, [metadata, isEnabled, store, handlersToSkip, clearStylePreset]);

  return {
    recall,
    isEnabled,
  };
};
