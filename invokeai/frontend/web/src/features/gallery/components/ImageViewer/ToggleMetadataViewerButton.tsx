import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectShouldShowImageDetails, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import { setShouldShowImageDetails } from 'features/ui/store/uiSlice';
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

  const shouldShowImageDetails = useAppSelector(selectShouldShowImageDetails);
  const imageDTO = useAppSelector(selectLastSelectedItem);
  const { t } = useTranslation();

  const toggleMetadataViewer = useCallback(() => {
    dispatch(setShouldShowImageDetails(!shouldShowImageDetails));
  }, [dispatch, shouldShowImageDetails]);

  useRegisteredHotkeys({
    id: 'toggleMetadata',
    category: 'viewer',
    callback: toggleMetadataViewer,
    dependencies: [imageDTO, shouldShowImageDetails],
  });

  return (
    <IconButton
      icon={<PiInfoBold />}
      tooltip={`${t('parameters.info')} (I)`}
      aria-label={`${t('parameters.info')} (I)`}
      onClick={toggleMetadataViewer}
      variant="link"
      alignSelf="stretch"
      colorScheme={shouldShowImageDetails ? 'invokeBlue' : 'base'}
      data-testid="toggle-show-metadata-button"
      isDisabled={isDisabledOverride}
    />
  );
});

ToggleMetadataViewerButton.displayName = 'ToggleMetadataViewerButton';
