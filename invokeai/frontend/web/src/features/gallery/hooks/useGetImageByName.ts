import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { ResourceOrigin } from 'services/api';
import { selectResultsEntities } from '../store/resultsSlice';
import { selectUploadsEntities } from '../store/uploadsSlice';

const useGetImageByNameSelector = createSelector(
  [selectResultsEntities, selectUploadsEntities],
  (allResults, allUploads) => {
    return { allResults, allUploads };
  }
);

const useGetImageByNameAndOrigin = () => {
  const { allResults, allUploads } = useAppSelector(useGetImageByNameSelector);
  return (name: string, origin: ResourceOrigin) => {
    if (origin === 'internal') {
      const resultImagesResult = allResults[name];
      if (resultImagesResult) {
        return resultImagesResult;
      }
    }

    if (origin === 'external') {
      const userImagesResult = allUploads[name];
      if (userImagesResult) {
        return userImagesResult;
      }
    }
  };
};

export default useGetImageByNameAndOrigin;
