import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { setShouldShowGallery } from 'features/gallery/store/gallerySlice';
import { setShouldShowParametersPanel } from 'features/ui/store/uiSlice';
import { useHotkeys } from 'react-hotkeys-hook';
import { MdPhotoLibrary } from 'react-icons/md';
import { floatingSelector } from './FloatingParametersPanelButtons';

const FloatingGalleryButton = () => {
  const dispatch = useAppDispatch();
  const {
    shouldShowGallery,
    shouldShowGalleryButton,
    shouldPinGallery,
    shouldShowParametersPanel,
    shouldPinParametersPanel,
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
      if (shouldShowGallery || shouldShowParametersPanel) {
        dispatch(setShouldShowParametersPanel(false));
        dispatch(setShouldShowGallery(false));
      } else {
        dispatch(setShouldShowParametersPanel(true));
        dispatch(setShouldShowGallery(true));
      }
      if (shouldPinGallery || shouldPinParametersPanel)
        setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    },
    [shouldShowGallery, shouldShowParametersPanel]
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
