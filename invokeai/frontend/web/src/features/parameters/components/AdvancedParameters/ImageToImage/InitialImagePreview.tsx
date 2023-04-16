import { Flex, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import SelectImagePlaceholder from 'common/components/SelectImagePlaceholder';
import { useGetUrl } from 'common/util/getUrl';
import useGetImageByNameAndType from 'features/gallery/hooks/useGetImageByName';
import { selectResultsById } from 'features/gallery/store/resultsSlice';
import {
  clearInitialImage,
  initialImageSelected,
} from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash';
import { DragEvent, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { ImageType } from 'services/api';

const initialImagePreviewSelector = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { initialImage } = state.generation;
    const image = selectResultsById(state, initialImage as string);

    return {
      initialImage: image,
    };
  },
  { memoizeOptions: { resultEqualityCheck: isEqual } }
);

const InitialImagePreview = () => {
  const { initialImage } = useAppSelector(initialImagePreviewSelector);
  const { getUrl } = useGetUrl();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const getImageByNameAndType = useGetImageByNameAndType();

  const onError = () => {
    dispatch(
      addToast({
        title: t('toast.parametersFailed'),
        description: t('toast.parametersFailedDesc'),
        status: 'error',
        isClosable: true,
      })
    );
    dispatch(clearInitialImage());
  };

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      const name = e.dataTransfer.getData('invokeai/imageName');
      const type = e.dataTransfer.getData('invokeai/imageType') as ImageType;

      if (!name || !type) {
        return;
      }

      const image = getImageByNameAndType(name, type);

      if (!image) {
        return;
      }

      dispatch(initialImageSelected(image.name));
    },
    [getImageByNameAndType, dispatch]
  );

  return (
    <Flex
      sx={{
        height: '100%',
        width: '100%',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onDrop={handleDrop}
    >
      <Image
        sx={{
          fit: 'contain',
          borderRadius: 'base',
        }}
        src={getUrl(initialImage?.url)}
        onError={onError}
        fallback={<SelectImagePlaceholder />}
      />
    </Flex>
  );
};

export default InitialImagePreview;
