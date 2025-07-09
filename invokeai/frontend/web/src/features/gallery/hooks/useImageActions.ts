import { useStore } from '@nanostores/react';
import { adHocPostProcessingRequested } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { MetadataHandlers, MetadataUtils } from 'features/metadata/parsing';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import {
  activeStylePresetIdChanged,
  selectStylePresetActivePresetId,
} from 'features/stylePresets/store/stylePresetSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

export const useImageActions = (imageDTO: ImageDTO | null) => {
  const store = useAppStore();
  const { t } = useTranslation();
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);
  const isStaging = useAppSelector(selectIsStaging);
  const activeTab = useAppSelector(selectActiveTab);
  const { metadata } = useDebouncedMetadata(imageDTO?.image_name ?? null);
  const [hasMetadata, setHasMetadata] = useState(false);
  const [hasSeed, setHasSeed] = useState(false);
  const [hasPrompts, setHasPrompts] = useState(false);
  const hasTemplates = useStore($hasTemplates);
  const deleteImageModal = useDeleteImageModalApi();
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');
  const isUpscalingEnabled = useFeatureStatus('upscaling');

  const isCanvasTabAndStaging = useMemo(() => activeTab === 'canvas' && isStaging, [activeTab, isStaging]);

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

    // When we are staging and on canvas, the bbox is locked - we cannot recall width and height
    const skip = activeTab === 'canvas' && isStaging ? [MetadataHandlers.Width, MetadataHandlers.Height] : undefined;
    MetadataUtils.recallAll(metadata, store, skip);
    clearStylePreset();
  }, [imageDTO, metadata, activeTab, isStaging, store, clearStylePreset]);

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

  const isEnabledLoadWorkflow = useMemo(() => {
    if (!imageDTO) {
      return false;
    }
    if (!imageDTO.has_workflow) {
      return false;
    }
    if (!hasTemplates) {
      return false;
    }
    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }
    return true;
  }, [hasTemplates, imageDTO, isGalleryFocused, isViewerFocused]);
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();
  const loadWorkflow = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isEnabledLoadWorkflow) {
      return;
    }

    loadWorkflowWithDialog({ type: 'image', data: imageDTO.image_name });
  }, [imageDTO, isEnabledLoadWorkflow, loadWorkflowWithDialog]);

  const isEnabledRecallSize = useMemo(() => {
    if (!imageDTO) {
      return;
    }
    if (activeTab === 'canvas' && isStaging) {
      return false;
    }
    if (activeTab !== 'canvas' && activeTab !== 'generate') {
      return false;
    }
    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }
    return true;
  }, [imageDTO, activeTab, isStaging, isGalleryFocused, isViewerFocused]);

  const recallSize = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isEnabledRecallSize) {
      return;
    }
    MetadataUtils.recallDimensions(imageDTO, store);
  }, [imageDTO, isEnabledRecallSize, store]);

  const isEnabledUpscale = useMemo(() => {
    if (!imageDTO) {
      return;
    }
    if (!isUpscalingEnabled) {
      return false;
    }
    if (activeTab === 'canvas' && isStaging) {
      return false;
    }
    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }
    return true;
  }, [imageDTO, isUpscalingEnabled, activeTab, isStaging, isGalleryFocused, isViewerFocused]);

  const upscale = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isEnabledUpscale) {
      return;
    }
    store.dispatch(adHocPostProcessingRequested({ imageDTO }));
  }, [imageDTO, isEnabledUpscale, store]);

  const isEnabledDelete = useMemo(() => {
    if (!imageDTO) {
      return;
    }
    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }
    return true;
  }, [imageDTO, isGalleryFocused, isViewerFocused]);
  const _delete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isEnabledDelete) {
      return;
    }
    deleteImageModal.delete([imageDTO.image_name]);
  }, [deleteImageModal, imageDTO, isEnabledDelete]);

  return {
    recallAll: {
      run: recallAll,
      isEnabled: hasMetadata,
    },
    remix: {
      run: remix,
      isEnabled: hasMetadata,
    },
    recallSeed: {
      run: recallSeed,
      isEnabled: hasSeed,
    },
    recallPrompts: {
      run: recallPrompts,
      isEnabled: hasPrompts,
    },
    createAsPreset,
    loadWorkflow: {
      run: loadWorkflow,
      isEnabled: imageDTO?.has_workflow ?? false,
    },
    recallSize: {
      run: recallSize,
      isEnabled: isEnabledRecallSize,
    },
    upscale: {
      run: upscale,
      isEnabled: isEnabledUpscale,
    },
    delete: {
      run: _delete,
      isEnabled: isEnabledDelete,
    },
  };
};
