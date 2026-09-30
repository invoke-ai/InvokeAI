import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useImageViewerContext } from 'features/gallery/components/ImageViewer/context';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useSharedInfoHotkeyTarget } from 'features/system/components/HotkeysModal/useSharedInfoHotkeyTarget';
import { selectShouldShowItemDetails, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import { setShouldShowItemDetails } from 'features/ui/store/uiSlice';
import { useCallback, useMemo } from 'react';

export const useToggleMetadataHotkey = () => {
  const dispatch = useAppDispatch();
  const ctx = useImageViewerContext();
  const hasProgressImage = useStore(ctx.$hasProgressImage);
  const isTemporarilyShowingSelectedImage = useStore(ctx.$isTemporarilyShowingSelectedImage);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const shouldShowItemDetails = useAppSelector(selectShouldShowItemDetails);

  const isDisabledOverride = hasProgressImage && shouldShowProgressInViewer && !isTemporarilyShowingSelectedImage;
  const hotkeyTarget = useSharedInfoHotkeyTarget();

  const toggleMetadataViewer = useCallback(() => {
    dispatch(setShouldShowItemDetails(!shouldShowItemDetails));
  }, [dispatch, shouldShowItemDetails]);
  const hotkeyOptions = useMemo(
    () => ({ enabled: hotkeyTarget === 'metadata' && !isDisabledOverride, preventDefault: true }),
    [hotkeyTarget, isDisabledOverride]
  );

  useRegisteredHotkeys({
    id: 'toggleMetadata',
    category: 'viewer',
    callback: toggleMetadataViewer,
    options: hotkeyOptions,
    dependencies: [toggleMetadataViewer, hotkeyTarget, isDisabledOverride],
  });
};
