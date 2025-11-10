import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import type { TabName } from 'features/controlLayers/store/types';
import { ImageMetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

import { useClearStylePresetWithToast } from './useClearStylePresetWithToast';

const ALLOWED_TABS: TabName[] = ['canvas', 'generate', 'upscaling'];

export const useRecallPrompts = (imageDTO: ImageDTO) => {
  const store = useAppStore();
  const tab = useAppSelector(selectActiveTab);
  const clearStylePreset = useClearStylePresetWithToast();
  const [hasPrompts, setHasPrompts] = useState(false);

  const { metadata, isLoading } = useDebouncedMetadata(imageDTO.image_name);

  useEffect(() => {
    const parse = async () => {
      try {
        const result = await MetadataUtils.hasMetadataByHandlers({
          handlers: [ImageMetadataHandlers.PositivePrompt, ImageMetadataHandlers.NegativePrompt],
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
    if (isLoading) {
      return false;
    }

    if (!ALLOWED_TABS.includes(tab)) {
      return false;
    }

    if (!hasPrompts) {
      return false;
    }

    return true;
  }, [hasPrompts, isLoading, tab]);

  const recall = useCallback(() => {
    if (!metadata) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    MetadataUtils.recallImagePrompts(metadata, store);
    clearStylePreset();
  }, [metadata, isEnabled, store, clearStylePreset]);

  return {
    recall,
    isEnabled,
  };
};
