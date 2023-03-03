import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { setShouldShowGallery } from 'features/gallery/store/gallerySlice';
import { MdPhotoLibrary } from 'react-icons/md';
import { floatingSelector } from './FloatingParametersPanelButtons';

const FloatingGalleryButton = () => {
  const dispatch = useAppDispatch();
  const { shouldShowGalleryButton, shouldPinGallery } =
    useAppSelector(floatingSelector);

  const handleShowGallery = () => {
    dispatch(setShouldShowGallery(true));
    if (shouldPinGallery) {
      dispatch(setDoesCanvasNeedScaling(true));
    }
  };

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
