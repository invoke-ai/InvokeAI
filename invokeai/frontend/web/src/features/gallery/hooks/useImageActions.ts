import { useStore } from '@nanostores/react';
import { adHocPostProcessingRequested } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import {
  handlers,
  parseAndRecallAllMetadata,
  parseAndRecallImageDimensions,
  parseAndRecallPrompts,
} from 'features/metadata/util/handlers';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import {
  activeStylePresetIdChanged,
  selectStylePresetActivePresetId,
} from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useGetAndLoadEmbeddedWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadEmbeddedWorkflow';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

export const useImageActions = (imageDTO: ImageDTO) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);
  const isStaging = useAppSelector(selectIsStaging);
  const activeTabName = useAppSelector(selectActiveTab);
  const { metadata } = useDebouncedMetadata(imageDTO.image_name);
  const [hasMetadata, setHasMetadata] = useState(false);
  const [hasSeed, setHasSeed] = useState(false);
  const [hasPrompts, setHasPrompts] = useState(false);
  const hasTemplates = useStore($hasTemplates);

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

        // Need to catch all of these to avoid unhandled promise rejections bubbling up to instrumented error handlers
        const promptParseResults = await Promise.allSettled([
          handlers.positivePrompt.parse(metadata).catch(() => {}),
          handlers.negativePrompt.parse(metadata).catch(() => {}),
          handlers.sdxlPositiveStylePrompt.parse(metadata).catch(() => {}),
          handlers.sdxlNegativeStylePrompt.parse(metadata).catch(() => {}),
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
    if (!metadata) {
      return;
    }
    parseAndRecallAllMetadata(metadata, activeTabName === 'canvas', isStaging ? ['width', 'height'] : []);
    clearStylePreset();
  }, [metadata, activeTabName, isStaging, clearStylePreset]);

  const remix = useCallback(() => {
    if (!metadata) {
      return;
    }
    // Recalls all metadata parameters except seed
    parseAndRecallAllMetadata(metadata, activeTabName === 'canvas', ['seed']);
    clearStylePreset();
  }, [activeTabName, metadata, clearStylePreset]);

  const recallSeed = useCallback(() => {
    if (!metadata) {
      return;
    }
    handlers.seed
      .parse(metadata)
      .then((seed) => {
        handlers.seed.recall?.(seed, true);
      })
      .catch(() => {
        // no-op, the toast will show the error
      });
  }, [metadata]);

  const recallPrompts = useCallback(() => {
    if (!metadata) {
      return;
    }
    parseAndRecallPrompts(metadata);
    clearStylePreset();
  }, [metadata, clearStylePreset]);

  const createAsPreset = useCallback(async () => {
    if (!metadata) {
      return;
    }
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
  }, [metadata, imageDTO]);

  const [getAndLoadEmbeddedWorkflow] = useGetAndLoadEmbeddedWorkflow();

  const loadWorkflow = useCallback(() => {
    if (!imageDTO.has_workflow || !hasTemplates) {
      return;
    }
    getAndLoadEmbeddedWorkflow(imageDTO.image_name);
  }, [getAndLoadEmbeddedWorkflow, hasTemplates, imageDTO.has_workflow, imageDTO.image_name]);

  const recallSize = useCallback(() => {
    if (isStaging) {
      return;
    }
    parseAndRecallImageDimensions(imageDTO);
  }, [imageDTO, isStaging]);

  const upscale = useCallback(() => {
    dispatch(adHocPostProcessingRequested({ imageDTO }));
  }, [dispatch, imageDTO]);

  const _delete = useCallback(() => {
    dispatch(imagesToDeleteSelected([imageDTO]));
  }, [dispatch, imageDTO]);

  return {
    hasMetadata,
    hasSeed,
    hasPrompts,
    recallAll,
    remix,
    recallSeed,
    recallPrompts,
    createAsPreset,
    loadWorkflow,
    hasWorkflow: imageDTO.has_workflow,
    recallSize,
    upscale,
    delete: _delete,
  };
};
