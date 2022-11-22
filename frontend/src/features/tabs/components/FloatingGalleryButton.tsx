import { MdPhotoLibrary } from 'react-icons/md';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { setShouldShowGallery } from 'features/gallery/store/gallerySlice';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { useHotkeys } from 'react-hotkeys-hook';
import { floatingSelector } from './FloatingOptionsPanelButtons';
import { setShouldShowOptionsPanel } from 'features/options/store/optionsSlice';

const FloatingGalleryButton = () => {
  const dispatch = useAppDispatch();
  const {
    shouldShowGallery,
    shouldShowGalleryButton,
    shouldPinGallery,
    shouldShowOptionsPanel,
    shouldPinOptionsPanel,
  } = useAppSelector(floatingSelector);

  const handleShowGallery = () => {
    dispatch(setShouldShowGallery(true));
    if (shouldPinGallery) {
      dispatch(setDoesCanvasNeedScaling(true));
    }
  };

  useHotkeys(
    'f',
    () => {
      if (shouldShowGallery || shouldShowOptionsPanel) {
        dispatch(setShouldShowOptionsPanel(false));
        dispatch(setShouldShowGallery(false));
      } else {
        dispatch(setShouldShowOptionsPanel(true));
        dispatch(setShouldShowGallery(true));
      }
      if (shouldPinGallery || shouldPinOptionsPanel)
        setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    },
    [shouldShowGallery, shouldShowOptionsPanel]
  );

  return shouldShowGalleryButton ? (
    <IAIIconButton
      tooltip="Show Gallery (G)"
      tooltipProps={{ placement: 'top' }}
      aria-label="Show Gallery"
      styleClass="floating-show-hide-button right show-hide-button-gallery"
      onClick={handleShowGallery}
    >
      <MdPhotoLibrary />
    </IAIIconButton>
  ) : null;
};

export default FloatingGalleryButton;
