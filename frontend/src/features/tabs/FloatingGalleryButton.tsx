import { MdPhotoLibrary } from 'react-icons/md';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { setShouldShowGallery } from 'features/gallery/gallerySlice';
import { setDoesCanvasNeedScaling } from 'features/canvas/canvasSlice';

const FloatingGalleryButton = () => {
  const dispatch = useAppDispatch();
  const shouldPinGallery = useAppSelector(
    (state: RootState) => state.gallery.shouldPinGallery
  );

  const handleShowGallery = () => {
    dispatch(setShouldShowGallery(true));
    if (shouldPinGallery) {
      dispatch(setDoesCanvasNeedScaling(true));
    }
  };

  return (
    <IAIIconButton
      tooltip="Show Gallery (G)"
      tooltipProps={{ placement: 'top' }}
      aria-label="Show Gallery"
      styleClass="floating-show-hide-button right show-hide-button-gallery"
      onClick={handleShowGallery}
    >
      <MdPhotoLibrary />
    </IAIIconButton>
  );
};

export default FloatingGalleryButton;
