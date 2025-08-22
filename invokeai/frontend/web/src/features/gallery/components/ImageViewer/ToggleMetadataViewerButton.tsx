import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectShouldShowItemDetails, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import { setShouldShowItemDetails } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';

import { useImageViewerContext } from './context';

export const ToggleMetadataViewerButton = memo(() => {
  const dispatch = useAppDispatch();
  const ctx = useImageViewerContext();
  const hasProgressImage = useStore(ctx.$hasProgressImage);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);

  const isDisabledOverride = hasProgressImage && shouldShowProgressInViewer;

  const shouldShowItemDetails = useAppSelector(selectShouldShowItemDetails);
  const imageDTO = useAppSelector(selectLastSelectedItem);
  const { t } = useTranslation();

  const toggleMetadataViewer = useCallback(() => {
    dispatch(setShouldShowItemDetails(!shouldShowItemDetails));
  }, [dispatch, shouldShowItemDetails]);

  useRegisteredHotkeys({
    id: 'toggleMetadata',
    category: 'viewer',
    callback: toggleMetadataViewer,
    dependencies: [imageDTO, shouldShowItemDetails],
  });

  return (
    <IconButton
      icon={<PiInfoBold />}
      tooltip={`${t('parameters.info')} (I)`}
      aria-label={`${t('parameters.info')} (I)`}
      onClick={toggleMetadataViewer}
      variant="link"
      alignSelf="stretch"
      colorScheme={shouldShowItemDetails ? 'invokeBlue' : 'base'}
      data-testid="toggle-show-metadata-button"
      isDisabled={isDisabledOverride}
    />
  );
});

ToggleMetadataViewerButton.displayName = 'ToggleMetadataViewerButton';
