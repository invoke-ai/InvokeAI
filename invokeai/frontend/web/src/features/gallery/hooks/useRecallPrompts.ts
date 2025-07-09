import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { MetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

import { useClearStylePresetWithToast } from './useClearStylePresetWithToast';

export const useRecallPrompts = (imageDTO?: ImageDTO | null) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');
  const clearStylePreset = useClearStylePresetWithToast();
  const [hasPrompts, setHasPrompts] = useState(false);

  const { metadata } = useDebouncedMetadata(imageDTO?.image_name);

  useEffect(() => {
    const parse = async () => {
      try {
        const result = await MetadataUtils.hasMetadataByHandlers({
          handlers: [
            MetadataHandlers.PositivePrompt,
            MetadataHandlers.NegativePrompt,
            MetadataHandlers.PositiveStylePrompt,
            MetadataHandlers.NegativeStylePrompt,
          ],
          metadata,
          store,
          require: 'some',
        });
        setHasPrompts(result);
      } catch {
        setHasPrompts(false);
      }
    };

    parse();
  }, [metadata, store]);

  const isEnabled = useMemo(() => {
    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }

    if (tab !== 'canvas' && tab !== 'generate') {
      return false;
    }

    if (!hasPrompts) {
      return false;
    }

    return true;
  }, [hasPrompts, isGalleryFocused, isViewerFocused, tab]);

  const recall = useCallback(() => {
    if (!metadata) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    MetadataUtils.recallPrompts(metadata, store);
    clearStylePreset();
  }, [metadata, isEnabled, store, clearStylePreset]);

  return {
    recall,
    isEnabled,
  };
};
