import { useAppStore } from 'app/store/storeHooks';
import { MetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

export const useCreateStylePresetFromMetadata = (imageDTO?: ImageDTO | null) => {
  const store = useAppStore();
  const [hasPrompts, setHasPrompts] = useState(false);

  const { metadata } = useDebouncedMetadata(imageDTO?.image_name);

  useEffect(() => {
    MetadataUtils.hasMetadataByHandlers({
      handlers: [MetadataHandlers.PositivePrompt, MetadataHandlers.NegativePrompt],
      metadata,
      store,
      require: 'some',
    })
      .then((result) => {
        setHasPrompts(result);
      })
      .catch(() => {
        setHasPrompts(false);
      });
  }, [metadata, store]);

  const isEnabled = useMemo(() => {
    if (!imageDTO) {
      return false;
    }
    if (!hasPrompts) {
      return false;
    }
    return true;
  }, [hasPrompts, imageDTO]);

  const create = useCallback(async () => {
    if (!imageDTO) {
      return;
    }
    if (!metadata) {
      return;
    }
    if (!isEnabled) {
      return;
    }

    let positivePrompt: string;
    let negativePrompt: string;

    try {
      positivePrompt = await MetadataHandlers.PositivePrompt.parse(metadata, store);
    } catch (error) {
      positivePrompt = '';
    }
    try {
      negativePrompt = (await MetadataHandlers.NegativePrompt.parse(metadata, store)) ?? '';
    } catch (error) {
      negativePrompt = '';
    }

    $stylePresetModalState.set({
      prefilledFormData: {
        name: '',
        positivePrompt,
        negativePrompt,
        imageUrl: imageDTO.image_url,
        type: 'user',
      },
      updatingStylePresetId: null,
      isModalOpen: true,
    });
  }, [imageDTO, isEnabled, metadata, store]);

  return {
    create,
    isEnabled,
  };
};
