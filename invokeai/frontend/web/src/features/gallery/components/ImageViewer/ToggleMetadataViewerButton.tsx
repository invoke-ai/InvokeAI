import { IconButton } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectShouldShowImageDetails } from 'features/ui/store/uiSelectors';
import { setShouldShowImageDetails } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

export const ToggleMetadataViewerButton = memo(() => {
  const dispatch = useAppDispatch();
  const shouldShowImageDetails = useAppSelector(selectShouldShowImageDetails);
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const { t } = useTranslation();

  const { currentData: imageDTO } = useGetImageDTOQuery(lastSelectedImage?.image_name ?? skipToken);

  const toggleMetadataViewer = useCallback(
    () => dispatch(setShouldShowImageDetails(!shouldShowImageDetails)),
    [dispatch, shouldShowImageDetails]
  );

  useRegisteredHotkeys({
    id: 'toggleMetadata',
    category: 'viewer',
    callback: toggleMetadataViewer,
    options: { enabled: Boolean(imageDTO) },
    dependencies: [imageDTO, shouldShowImageDetails],
  });

  return (
    <IconButton
      icon={<PiInfoBold />}
      tooltip={`${t('parameters.info')} (I)`}
      aria-label={`${t('parameters.info')} (I)`}
      onClick={toggleMetadataViewer}
      isDisabled={!imageDTO}
      variant="outline"
      colorScheme={shouldShowImageDetails ? 'invokeBlue' : 'base'}
      data-testid="toggle-show-metadata-button"
    />
  );
});

ToggleMetadataViewerButton.displayName = 'ToggleMetadataViewerButton';
