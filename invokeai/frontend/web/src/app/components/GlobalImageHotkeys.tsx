import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useLoadWorkflow } from 'features/gallery/hooks/useLoadWorkflow';
import { useRecallAll } from 'features/gallery/hooks/useRecallAll';
import { useRecallDimensions } from 'features/gallery/hooks/useRecallDimensions';
import { useRecallPrompts } from 'features/gallery/hooks/useRecallPrompts';
import { useRecallRemix } from 'features/gallery/hooks/useRecallRemix';
import { useRecallSeed } from 'features/gallery/hooks/useRecallSeed';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useImageDTO } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

export const GlobalImageHotkeys = memo(() => {
  useAssertSingleton('GlobalImageHotkeys');
  const imageName = useAppSelector(selectLastSelectedImage);
  const imageDTO = useImageDTO(imageName);

  if (!imageDTO) {
    return null;
  }

  return <GlobalImageHotkeysInternal imageDTO={imageDTO} />;
});

GlobalImageHotkeys.displayName = 'GlobalImageHotkeys';

const GlobalImageHotkeysInternal = memo(({ imageDTO }: { imageDTO: ImageDTO }) => {
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
    options: { enabled: loadWorkflow.isEnabled },
    dependencies: [loadWorkflow],
  });

  useRegisteredHotkeys({
    id: 'recallAll',
    category: 'viewer',
    callback: recallAll.recall,
    options: { enabled: recallAll.isEnabled },
    dependencies: [recallAll],
  });

  useRegisteredHotkeys({
    id: 'recallSeed',
    category: 'viewer',
    callback: recallSeed.recall,
    options: { enabled: recallSeed.isEnabled },
    dependencies: [recallSeed],
  });

  useRegisteredHotkeys({
    id: 'recallPrompts',
    category: 'viewer',
    callback: recallPrompts.recall,
    options: { enabled: recallPrompts.isEnabled },
    dependencies: [recallPrompts],
  });

  useRegisteredHotkeys({
    id: 'remix',
    category: 'viewer',
    callback: recallRemix.recall,
    options: { enabled: recallRemix.isEnabled },
    dependencies: [recallRemix],
  });

  useRegisteredHotkeys({
    id: 'useSize',
    category: 'viewer',
    callback: recallDimensions.recall,
    options: { enabled: recallDimensions.isEnabled },
    dependencies: [recallDimensions],
  });

  return null;
});

GlobalImageHotkeysInternal.displayName = 'GlobalImageHotkeysInternal';
