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
      onClick={handleShowGallery}
      sx={{
        pos: 'absolute',
        top: '50%',
        transform: 'translate(0, -50%)',
        zIndex: 31,
        p: 0,
        insetInlineEnd: 0,
        px: 3,
        h: 48,
        w: 8,
        borderStartEndRadius: 0,
        borderEndEndRadius: 0,
      }}
    >
      <MdPhotoLibrary />
    </IAIIconButton>
  ) : null;
};

export default FloatingGalleryButton;
