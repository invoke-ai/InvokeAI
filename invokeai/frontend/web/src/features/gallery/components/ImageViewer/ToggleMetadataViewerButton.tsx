import { IconButton } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { setShouldShowImageDetails } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

export const ToggleMetadataViewerButton = memo(() => {
  const dispatch = useAppDispatch();
  const shouldShowImageDetails = useAppSelector((s) => s.ui.shouldShowImageDetails);
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const toaster = useAppToaster();
  const { t } = useTranslation();

  const { currentData: imageDTO } = useGetImageDTOQuery(lastSelectedImage?.image_name ?? skipToken);

  const toggleMetadataViewer = useCallback(
    () => dispatch(setShouldShowImageDetails(!shouldShowImageDetails)),
    [dispatch, shouldShowImageDetails]
  );

  useHotkeys('i', toggleMetadataViewer, { enabled: Boolean(imageDTO) }, [imageDTO, shouldShowImageDetails, toaster]);

  return (
    <IconButton
      icon={<PiInfoBold />}
      tooltip={`${t('parameters.info')} (I)`}
      aria-label={`${t('parameters.info')} (I)`}
      onClick={toggleMetadataViewer}
      isDisabled={!imageDTO}
      variant="outline"
      colorScheme={shouldShowImageDetails ? 'invokeBlue' : 'base'}
    />
  );
});

ToggleMetadataViewerButton.displayName = 'ToggleMetadataViewerButton';
