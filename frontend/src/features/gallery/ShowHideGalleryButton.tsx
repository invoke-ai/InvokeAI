import { useHotkeys } from 'react-hotkeys-hook';
import { MdPhotoLibrary } from 'react-icons/md';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import IAIIconButton from '../../common/components/IAIIconButton';
import { setShouldShowGallery } from '../gallery/gallerySlice';
import { selectNextImage, selectPrevImage } from './gallerySlice';

const ShowHideGalleryButton = () => {
  const dispatch = useAppDispatch();

  const { shouldPinGallery, shouldShowGallery } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const handleShowGalleryToggle = () => {
    dispatch(setShouldShowGallery(!shouldShowGallery));
  };

  // useHotkeys(
  //   'g',
  //   () => {
  //     handleShowGalleryToggle();
  //   },
  //   [shouldShowGallery]
  // );

  // useHotkeys(
  //   'left',
  //   () => {
  //     dispatch(selectPrevImage());
  //   },
  //   []
  // );

  // useHotkeys(
  //   'right',
  //   () => {
  //     dispatch(selectNextImage());
  //   },
  //   []
  // );

  return (
    <IAIIconButton
      tooltip="Show Gallery (G)"
      tooltipPlacement="top"
      aria-label="Show Gallery"
      onClick={handleShowGalleryToggle}
      styleClass="show-hide-gallery-button"
      onMouseOver={!shouldPinGallery ? handleShowGalleryToggle : undefined}
    >
      <MdPhotoLibrary />
    </IAIIconButton>
  );
};

export default ShowHideGalleryButton;
