import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { setShouldShowGallery } from 'features/ui/store/uiSlice';
import { isEqual } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdPhotoLibrary } from 'react-icons/md';
import { activeTabNameSelector, uiSelector } from '../store/uiSelectors';
import { NO_GALLERY_TABS } from './InvokeTabs';

const floatingGalleryButtonSelector = createSelector(
  [activeTabNameSelector, uiSelector],
  (activeTabName, ui) => {
    const { shouldPinGallery, shouldShowGallery } = ui;

    return {
      shouldPinGallery,
      shouldShowGalleryButton: NO_GALLERY_TABS.includes(activeTabName)
        ? false
        : !shouldShowGallery,
    };
  },
  { memoizeOptions: { resultEqualityCheck: isEqual } }
);

const FloatingGalleryButton = () => {
  const { t } = useTranslation();
  const { shouldPinGallery, shouldShowGalleryButton } = useAppSelector(
    floatingGalleryButtonSelector
  );
  const dispatch = useAppDispatch();

  const handleShowGallery = () => {
    dispatch(setShouldShowGallery(true));
    shouldPinGallery && dispatch(requestCanvasRescale());
  };

  return shouldShowGalleryButton ? (
    <IAIIconButton
      tooltip="Show Gallery (G)"
      tooltipProps={{ placement: 'top' }}
      aria-label={t('accessibility.showGallery')}
      onClick={handleShowGallery}
      sx={{
        pos: 'absolute',
        top: '50%',
        transform: 'translate(0, -50%)',
        p: 0,
        insetInlineEnd: 0,
        px: 3,
        h: 48,
        w: 8,
        borderStartEndRadius: 0,
        borderEndEndRadius: 0,
        shadow: '2xl',
      }}
    >
      <MdPhotoLibrary />
    </IAIIconButton>
  ) : null;
};

export default memo(FloatingGalleryButton);
