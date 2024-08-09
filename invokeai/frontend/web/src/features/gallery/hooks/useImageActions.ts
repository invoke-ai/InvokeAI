import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { handlers, parseAndRecallAllMetadata, parseAndRecallPrompts } from 'features/metadata/util/handlers';
import { isModalOpenChanged, prefilledFormDataChanged } from 'features/stylePresets/store/stylePresetModalSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback, useEffect, useState } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';

export const useImageActions = (image_name?: string) => {
  const activeTabName = useAppSelector(activeTabNameSelector);
  const { metadata, isLoading: isLoadingMetadata } = useDebouncedMetadata(image_name);
  const [hasMetadata, setHasMetadata] = useState(false);
  const [hasSeed, setHasSeed] = useState(false);
  const [hasPrompts, setHasPrompts] = useState(false);
  const dispatch = useAppDispatch();
  const { data: imageDTO } = useGetImageDTOQuery(image_name ?? skipToken);

  useEffect(() => {
    const parseMetadata = async () => {
      if (metadata) {
        setHasMetadata(true);
        try {
          await handlers.seed.parse(metadata);
          setHasSeed(true);
        } catch {
          setHasSeed(false);
        }

        const promptParseResults = await Promise.allSettled([
          handlers.positivePrompt.parse(metadata),
          handlers.negativePrompt.parse(metadata),
          handlers.sdxlPositiveStylePrompt.parse(metadata),
          handlers.sdxlNegativeStylePrompt.parse(metadata),
        ]);
        if (promptParseResults.some((result) => result.status === 'fulfilled')) {
          setHasPrompts(true);
        } else {
          setHasPrompts(false);
        }
      } else {
        setHasMetadata(false);
        setHasSeed(false);
        setHasPrompts(false);
      }
    };
    parseMetadata();
  }, [metadata]);

  const recallAll = useCallback(() => {
    parseAndRecallAllMetadata(metadata, activeTabName === 'generation');
  }, [activeTabName, metadata]);

  const remix = useCallback(() => {
    // Recalls all metadata parameters except seed
    parseAndRecallAllMetadata(metadata, activeTabName === 'generation', ['seed']);
  }, [activeTabName, metadata]);

  const recallSeed = useCallback(() => {
    handlers.seed.parse(metadata).then((seed) => {
      handlers.seed.recall && handlers.seed.recall(seed, true);
    });
  }, [metadata]);

  const recallPrompts = useCallback(() => {
    parseAndRecallPrompts(metadata);
  }, [metadata]);

  const createAsPreset = useCallback(async () => {
    if (image_name && metadata && imageDTO) {
      const positivePrompt = await handlers.positivePrompt.parse(metadata);
      const negativePrompt = await handlers.negativePrompt.parse(metadata);

      dispatch(
        prefilledFormDataChanged({
          name: '',
          positivePrompt,
          negativePrompt,
          imageUrl: imageDTO.image_url,
        })
      );
      dispatch(isModalOpenChanged(true));
    }
  }, [image_name, metadata, dispatch, imageDTO]);

  return {
    recallAll,
    remix,
    recallSeed,
    recallPrompts,
    hasMetadata,
    hasSeed,
    hasPrompts,
    isLoadingMetadata,
    createAsPreset,
  };
};
