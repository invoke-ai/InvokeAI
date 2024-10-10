import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

export const GlobalImageHotkeys = memo(() => {
  useAssertSingleton('GlobalImageHotkeys');
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const { currentData: imageDTO } = useGetImageDTOQuery(lastSelectedImage?.image_name ?? skipToken);

  if (!imageDTO) {
    return null;
  }

  return <GlobalImageHotkeysInternal imageDTO={imageDTO} />;
});

GlobalImageHotkeys.displayName = 'GlobalImageHotkeys';

const GlobalImageHotkeysInternal = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');
  const imageActions = useImageActions(imageDTO);
  const isStaging = useAppSelector(selectIsStaging);
  const isUpscalingEnabled = useFeatureStatus('upscaling');

  useRegisteredHotkeys({
    id: 'loadWorkflow',
    category: 'viewer',
    callback: imageActions.loadWorkflow,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [imageActions.loadWorkflow, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'recallAll',
    category: 'viewer',
    callback: imageActions.recallAll,
    options: { enabled: !isStaging && (isGalleryFocused || isViewerFocused) },
    dependencies: [imageActions.recallAll, isStaging, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'recallSeed',
    category: 'viewer',
    callback: imageActions.recallSeed,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [imageActions.recallSeed, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'recallPrompts',
    category: 'viewer',
    callback: imageActions.recallPrompts,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [imageActions.recallPrompts, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'remix',
    category: 'viewer',
    callback: imageActions.remix,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [imageActions.remix, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'useSize',
    category: 'viewer',
    callback: imageActions.recallSize,
    options: { enabled: !isStaging && (isGalleryFocused || isViewerFocused) },
    dependencies: [imageActions.recallSize, isStaging, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'runPostprocessing',
    category: 'viewer',
    callback: imageActions.upscale,
    options: { enabled: isUpscalingEnabled && isViewerFocused },
    dependencies: [isUpscalingEnabled, imageDTO, isViewerFocused],
  });
  return null;
});

GlobalImageHotkeysInternal.displayName = 'GlobalImageHotkeysInternal';
