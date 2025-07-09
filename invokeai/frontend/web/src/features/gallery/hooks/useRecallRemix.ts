import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { MetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

import { useClearStylePresetWithToast } from './useClearStylePresetWithToast';

export const useRecallRemix = (imageDTO: ImageDTO) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const isStaging = useAppSelector(selectIsStaging);
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
    const _handlersToSkip = [MetadataHandlers.Seed];
    if (tab === 'canvas' && isStaging) {
      // When we are staging and on canvas, the bbox is locked - we cannot recall width and height
      _handlersToSkip.push(MetadataHandlers.Width, MetadataHandlers.Height);
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
    MetadataUtils.recallAll(metadata, store, handlersToSkip);
    clearStylePreset();
  }, [metadata, isEnabled, store, handlersToSkip, clearStylePreset]);

  return {
    recall,
    isEnabled,
  };
};
