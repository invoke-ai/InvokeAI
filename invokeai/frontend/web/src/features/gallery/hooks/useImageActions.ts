import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { handlers, parseAndRecallAllMetadata, parseAndRecallPrompts } from 'features/metadata/util/handlers';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import { activeStylePresetIdChanged } from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';

export const useImageActions = (image_name?: string) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const activeStylePresetId = useAppSelector((s) => s.stylePreset.activeStylePresetId);
  const activeTabName = useAppSelector(selectActiveTab);
  const { metadata, isLoading: isLoadingMetadata } = useDebouncedMetadata(image_name);
  const [hasMetadata, setHasMetadata] = useState(false);
  const [hasSeed, setHasSeed] = useState(false);
  const [hasPrompts, setHasPrompts] = useState(false);
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

  const clearStylePreset = useCallback(() => {
    if (activeStylePresetId) {
      dispatch(activeStylePresetIdChanged(null));
      toast({
        status: 'info',
        title: t('stylePresets.promptTemplateCleared'),
      });
    }
  }, [dispatch, activeStylePresetId, t]);

  const recallAll = useCallback(() => {
    parseAndRecallAllMetadata(metadata, activeTabName === 'generation');
    clearStylePreset();
  }, [activeTabName, metadata, clearStylePreset]);

  const remix = useCallback(() => {
    // Recalls all metadata parameters except seed
    parseAndRecallAllMetadata(metadata, activeTabName === 'generation', ['seed']);
    clearStylePreset();
  }, [activeTabName, metadata, clearStylePreset]);

  const recallSeed = useCallback(() => {
    handlers.seed.parse(metadata).then((seed) => {
      handlers.seed.recall && handlers.seed.recall(seed, true);
    });
  }, [metadata]);

  const recallPrompts = useCallback(() => {
    parseAndRecallPrompts(metadata);
    clearStylePreset();
  }, [metadata, clearStylePreset]);

  const createAsPreset = useCallback(async () => {
    if (image_name && metadata && imageDTO) {
      let positivePrompt;
      let negativePrompt;

      try {
        positivePrompt = await handlers.positivePrompt.parse(metadata);
      } catch (error) {
        positivePrompt = '';
      }
      try {
        negativePrompt = await handlers.negativePrompt.parse(metadata);
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
    }
  }, [image_name, metadata, imageDTO]);

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
