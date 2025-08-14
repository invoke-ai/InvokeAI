import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectHasModelCLIPSkip } from 'features/controlLayers/store/paramsSlice';
import { MetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { TabName } from 'features/ui/store/uiTypes';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

const ALLOWED_TABS: TabName[] = ['canvas', 'generate', 'upscaling'];

export const useRecallCLIPSkip = (imageDTO: ImageDTO) => {
  const store = useAppStore();
  const hasModelCLIPSkip = useAppSelector(selectHasModelCLIPSkip);
  const tab = useAppSelector(selectActiveTab);
  const [hasCLIPSkip, setHasCLIPSkip] = useState(false);

  const { metadata, isLoading } = useDebouncedMetadata(imageDTO.image_name);

  useEffect(() => {
    const parse = async () => {
      try {
        await MetadataHandlers.CLIPSkip.parse(metadata, store);
        setHasCLIPSkip(true);
      } catch {
        setHasCLIPSkip(false);
      }
    };

    if (!hasModelCLIPSkip) {
      setHasCLIPSkip(false);
      return;
    }

    parse();
  }, [metadata, store, hasModelCLIPSkip]);

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

    if (!hasCLIPSkip) {
      return false;
    }

    return true;
  }, [hasCLIPSkip, isLoading, metadata, tab]);

  const recall = useCallback(() => {
    if (!metadata) {
      return;
    }
    if (!isEnabled) {
      return;
    }
    MetadataUtils.recallByHandler({ metadata, handler: MetadataHandlers.CLIPSkip, store });
  }, [metadata, isEnabled, store]);

  return {
    recall,
    isEnabled,
  };
};
