import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store';
import { gallerySelector } from '../store/gallerySliceSelectors';

const selector = createSelector(gallerySelector, (gallery) => ({
  resultImages: gallery.categories.result.images,
  userImages: gallery.categories.user.images,
}));

const useGetImageByUuid = () => {
  const { resultImages, userImages } = useAppSelector(selector);

  return (uuid: string) => {
    const resultImagesResult = resultImages.find(
      (image) => image.uuid === uuid
    );
    if (resultImagesResult) {
      return resultImagesResult;
    }

    const userImagesResult = userImages.find((image) => image.uuid === uuid);
    if (userImagesResult) {
      return userImagesResult;
    }
  };
};

export default useGetImageByUuid;
