import { useStore } from '@nanostores/react';
import { adHocPostProcessingRequested } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { MetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import {
  activeStylePresetIdChanged,
  selectStylePresetActivePresetId,
} from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

export const useImageActions = (imageDTO: ImageDTO | null) => {
  const store = useAppStore();
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
          await MetadataHandlers.Seed.parse(metadata, store);
          setHasSeed(true);
        } catch {
          setHasSeed(false);
        }

        let hasPrompt = false;
        // Need to catch all of these to avoid unhandled promise rejections bubbling up to instrumented error handlers
        for (const handler of [
          MetadataHandlers.PositivePrompt,
          MetadataHandlers.NegativePrompt,
          MetadataHandlers.PositiveStylePrompt,
          MetadataHandlers.NegativeStylePrompt,
        ]) {
          try {
            await handler.parse(metadata, store);
            hasPrompt = true;
            break;
          } catch {
            // noop
          }
        }
        setHasPrompts(hasPrompt);
      } else {
        setHasMetadata(false);
        setHasSeed(false);
        setHasPrompts(false);
      }
    };
    parseMetadata();
  }, [metadata, store]);

  const clearStylePreset = useCallback(() => {
    if (activeStylePresetId) {
      store.dispatch(activeStylePresetIdChanged(null));
      toast({
        status: 'info',
        title: t('stylePresets.promptTemplateCleared'),
      });
    }
  }, [activeStylePresetId, store, t]);

  const recallAll = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!metadata) {
      return;
    }
    MetadataUtils.recallAll(metadata, store, isStaging ? [MetadataHandlers.Width, MetadataHandlers.Height] : []);
    clearStylePreset();
  }, [imageDTO, metadata, store, isStaging, clearStylePreset]);

  const remix = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!metadata) {
      return;
    }
    // Recalls all metadata parameters except seed
    MetadataUtils.recallAll(metadata, store, [MetadataHandlers.Seed]);
    clearStylePreset();
  }, [imageDTO, metadata, store, clearStylePreset]);

  const recallSeed = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!metadata) {
      return;
    }
    MetadataUtils.recallByHandler({ metadata, store, handler: MetadataHandlers.Seed });
  }, [imageDTO, metadata, store]);

  const recallPrompts = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!metadata) {
      return;
    }
    MetadataUtils.recallPrompts(metadata, store);
    clearStylePreset();
  }, [imageDTO, metadata, store, clearStylePreset]);

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
  }, [imageDTO, metadata, store]);

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
    MetadataUtils.recallDimensions(imageDTO, store);
  }, [imageDTO, isStaging, store]);

  const upscale = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    store.dispatch(adHocPostProcessingRequested({ imageDTO }));
  }, [imageDTO, store]);

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
