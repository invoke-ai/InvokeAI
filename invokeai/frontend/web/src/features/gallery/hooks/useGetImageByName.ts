import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { ImageType } from 'services/api';
import { selectResultsEntities } from '../store/resultsSlice';
import { selectUploadsEntities } from '../store/uploadsSlice';

const useGetImageByNameSelector = createSelector(
  [selectResultsEntities, selectUploadsEntities],
  (allResults, allUploads) => {
    return { allResults, allUploads };
  }
);

const useGetImageByNameAndType = () => {
  const { allResults, allUploads } = useAppSelector(useGetImageByNameSelector);

  return (name: string, type: ImageType) => {
    if (type === 'results') {
      const resultImagesResult = allResults[name];

      if (resultImagesResult) {
        return resultImagesResult;
      }
    }

    if (type === 'uploads') {
      const userImagesResult = allUploads[name];
      if (userImagesResult) {
        return userImagesResult;
      }
    }
  };
};

export default useGetImageByNameAndType;
