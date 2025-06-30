import { useStore } from '@nanostores/react';
import { adHocPostProcessingRequested } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
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
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

export const useImageActions = (imageDTO: ImageDTO | null) => {
  const { dispatch, getState } = useAppStore();
  const { t } = useTranslation();
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);
  const isStaging = useAppSelector(selectIsStaging);
  const { metadata } = useDebouncedMetadata(imageDTO?.image_name ?? null);
  const [hasMetadata, setHasMetadata] = useState(false);
  const [hasSeed, setHasSeed] = useState(false);
  const [hasPrompts, setHasPrompts] = useState(false);
  const hasTemplates = useStore($hasTemplates);
  const deleteImageModal = useDeleteImageModalApi();

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
    if (!imageDTO) {
      return;
    }
    if (!metadata) {
      return;
    }
    const activeTabName = selectActiveTab(getState());
    parseAndRecallAllMetadata(metadata, activeTabName === 'canvas', isStaging ? ['width', 'height'] : []);
    clearStylePreset();
  }, [imageDTO, metadata, getState, isStaging, clearStylePreset]);

  const remix = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!metadata) {
      return;
    }
    const activeTabName = selectActiveTab(getState());
    // Recalls all metadata parameters except seed
    parseAndRecallAllMetadata(metadata, activeTabName === 'canvas', ['seed']);
    clearStylePreset();
  }, [imageDTO, metadata, getState, clearStylePreset]);

  const recallSeed = useCallback(() => {
    if (!imageDTO) {
      return;
    }
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
  }, [imageDTO, metadata]);

  const recallPrompts = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!metadata) {
      return;
    }
    parseAndRecallPrompts(metadata);
    clearStylePreset();
  }, [imageDTO, metadata, clearStylePreset]);

  const createAsPreset = useCallback(async () => {
    if (!imageDTO) {
      return;
    }
    if (!metadata) {
      return;
    }
    let positivePrompt: string;
    let negativePrompt: string;

    try {
      positivePrompt = await handlers.positivePrompt.parse(metadata);
    } catch (error) {
      positivePrompt = '';
    }
    try {
      negativePrompt = (await handlers.negativePrompt.parse(metadata)) ?? '';
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

  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();

  const loadWorkflowFromImage = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!imageDTO.has_workflow || !hasTemplates) {
      return;
    }

    loadWorkflowWithDialog({ type: 'image', data: imageDTO.image_name });
  }, [hasTemplates, imageDTO, loadWorkflowWithDialog]);

  const recallSize = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (isStaging) {
      return;
    }
    parseAndRecallImageDimensions(imageDTO);
  }, [imageDTO, isStaging]);

  const upscale = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(adHocPostProcessingRequested({ imageDTO }));
  }, [dispatch, imageDTO]);

  const _delete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    deleteImageModal.delete([imageDTO.image_name]);
  }, [deleteImageModal, imageDTO]);

  return {
    hasMetadata,
    hasSeed,
    hasPrompts,
    recallAll,
    remix,
    recallSeed,
    recallPrompts,
    createAsPreset,
    loadWorkflow: loadWorkflowFromImage,
    hasWorkflow: imageDTO?.has_workflow ?? false,
    recallSize,
    upscale,
    delete: _delete,
  };
};
