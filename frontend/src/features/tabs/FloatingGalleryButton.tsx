import { MdPhotoLibrary } from 'react-icons/md';
import { useAppDispatch } from '../../app/store';
import IAIIconButton from '../../common/components/IAIIconButton';
import { setShouldShowGallery } from '../gallery/gallerySlice';

const FloatingGalleryButton = () => {
  const dispatch = useAppDispatch();

  const handleShowGallery = () => {
    dispatch(setShouldShowGallery(true));
  };

  return (
    <IAIIconButton
      tooltip="Show Gallery (G)"
      tooltipProps={{ placement: 'top' }}
      aria-label="Show Gallery"
      styleClass="floating-show-hide-button right"
      onMouseOver={handleShowGallery}
    >
      <MdPhotoLibrary />
    </IAIIconButton>
  );
};

export default FloatingGalleryButton;
