import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { MetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

export const useRecallSeed = (imageDTO?: ImageDTO | null) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const [hasSeed, setHasSeed] = useState(false);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');

  const { metadata } = useDebouncedMetadata(imageDTO?.image_name);

  useEffect(() => {
    const parse = async () => {
      try {
        await MetadataHandlers.Seed.parse(metadata, store);
        setHasSeed(true);
      } catch {
        setHasSeed(false);
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

    if (!metadata) {
      return false;
    }

    if (!hasSeed) {
      return false;
    }

    return true;
  }, [hasSeed, isGalleryFocused, isViewerFocused, metadata, tab]);

  const recall = useCallback(() => {
    if (!metadata) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    MetadataUtils.recallByHandler({ metadata, handler: MetadataHandlers.Seed, store });
  }, [metadata, isEnabled, store]);

  return {
    recall,
    isEnabled,
  };
};
