import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIIconButton from 'common/components/IAIIconButton';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { togglePinGalleryPanel } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { shouldPinGallery } = state.ui;

    return {
      shouldPinGallery,
    };
  },
  defaultSelectorOptions
);

const GalleryPinButton = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { shouldPinGallery } = useAppSelector(selector);

  const handleSetShouldPinGallery = () => {
    dispatch(togglePinGalleryPanel());
    dispatch(requestCanvasRescale());
  };
  return (
    <IAIIconButton
      size="sm"
      aria-label={t('gallery.pinGallery')}
      tooltip={`${t('gallery.pinGallery')} (Shift+G)`}
      onClick={handleSetShouldPinGallery}
      icon={shouldPinGallery ? <BsPinAngleFill /> : <BsPinAngle />}
    />
  );
};

export default GalleryPinButton;
