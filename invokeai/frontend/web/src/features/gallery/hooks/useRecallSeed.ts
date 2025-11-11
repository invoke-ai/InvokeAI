import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import type { TabName } from 'features/controlLayers/store/types';
import { ImageMetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

const ALLOWED_TABS: TabName[] = ['canvas', 'generate', 'upscaling'];

export const useRecallSeed = (imageDTO: ImageDTO) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const [hasSeed, setHasSeed] = useState(false);

  const { metadata, isLoading } = useDebouncedMetadata(imageDTO.image_name);

  useEffect(() => {
    const parse = async () => {
      try {
        await ImageMetadataHandlers.Seed.parse(metadata, store);
        setHasSeed(true);
      } catch {
        setHasSeed(false);
      }
    };

    parse();
  }, [metadata, store]);

  const isEnabled = useMemo(() => {
    if (isLoading) {
      return false;
    }

    if (!ALLOWED_TABS.includes(tab)) {
      return false;
    }

    if (!metadata) {
      return false;
    }

    if (!hasSeed) {
      return false;
    }

    return true;
  }, [hasSeed, isLoading, metadata, tab]);

  const recall = useCallback(() => {
    if (!metadata) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    MetadataUtils.recallByHandler({ metadata, handler: ImageMetadataHandlers.Seed, store });
  }, [metadata, isEnabled, store]);

  return {
    recall,
    isEnabled,
  };
};
