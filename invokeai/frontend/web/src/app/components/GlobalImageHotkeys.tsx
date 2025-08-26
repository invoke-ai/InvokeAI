import { useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useLoadWorkflow } from 'features/gallery/hooks/useLoadWorkflow';
import { useRecallAll } from 'features/gallery/hooks/useRecallAllImageMetadata';
import { useRecallDimensions } from 'features/gallery/hooks/useRecallDimensions';
import { useRecallPrompts } from 'features/gallery/hooks/useRecallPrompts';
import { useRecallRemix } from 'features/gallery/hooks/useRecallRemix';
import { useRecallSeed } from 'features/gallery/hooks/useRecallSeed';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useImageDTO } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

export const GlobalImageHotkeys = memo(() => {
  useAssertSingleton('GlobalImageHotkeys');
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const imageDTO = useImageDTO(lastSelectedItem?.type === 'image' ? lastSelectedItem.id : null);

  if (!imageDTO) {
    return null;
  }

  return <GlobalImageHotkeysInternal imageDTO={imageDTO} />;
});

GlobalImageHotkeys.displayName = 'GlobalImageHotkeys';

const GlobalImageHotkeysInternal = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');

  const isFocusOK = isGalleryFocused || isViewerFocused;

  const recallAll = useRecallAll(imageDTO);
  const recallRemix = useRecallRemix(imageDTO);
  const recallPrompts = useRecallPrompts(imageDTO);
  const recallSeed = useRecallSeed(imageDTO);
  const recallDimensions = useRecallDimensions(imageDTO);
  const loadWorkflow = useLoadWorkflow(imageDTO);

  useRegisteredHotkeys({
    id: 'loadWorkflow',
    category: 'viewer',
    callback: loadWorkflow.load,
    options: { enabled: loadWorkflow.isEnabled && isFocusOK },
    dependencies: [loadWorkflow, isFocusOK],
  });

  useRegisteredHotkeys({
    id: 'recallAll',
    category: 'viewer',
    callback: recallAll.recall,
    options: { enabled: recallAll.isEnabled && isFocusOK },
    dependencies: [recallAll, isFocusOK],
  });

  useRegisteredHotkeys({
    id: 'recallSeed',
    category: 'viewer',
    callback: recallSeed.recall,
    options: { enabled: recallSeed.isEnabled && isFocusOK },
    dependencies: [recallSeed, isFocusOK],
  });

  useRegisteredHotkeys({
    id: 'recallPrompts',
    category: 'viewer',
    callback: recallPrompts.recall,
    options: { enabled: recallPrompts.isEnabled && isFocusOK },
    dependencies: [recallPrompts, isFocusOK],
  });

  useRegisteredHotkeys({
    id: 'remix',
    category: 'viewer',
    callback: recallRemix.recall,
    options: { enabled: recallRemix.isEnabled && isFocusOK },
    dependencies: [recallRemix, isFocusOK],
  });

  useRegisteredHotkeys({
    id: 'useSize',
    category: 'viewer',
    callback: recallDimensions.recall,
    options: { enabled: recallDimensions.isEnabled && isFocusOK },
    dependencies: [recallDimensions, isFocusOK],
  });

  return null;
});

GlobalImageHotkeysInternal.displayName = 'GlobalImageHotkeysInternal';
