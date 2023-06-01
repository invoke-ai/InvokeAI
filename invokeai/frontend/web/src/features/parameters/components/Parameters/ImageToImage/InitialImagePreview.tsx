import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGetUrl } from 'common/util/getUrl';
import {
  clearInitialImage,
  initialImageChanged,
} from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { configSelector } from '../../../../system/store/configSelectors';
import { useAppToaster } from 'app/components/Toaster';
import IAISelectableImage from 'features/controlNet/components/parameters/IAISelectableImage';
import { ImageDTO } from 'services/api';

const selector = createSelector(
  [generationSelector],
  (generation) => {
    const { initialImage } = generation;
    return {
      initialImage,
    };
  },
  defaultSelectorOptions
);

const InitialImagePreview = () => {
  const { initialImage } = useAppSelector(selector);
  const { shouldFetchImages } = useAppSelector(configSelector);
  const { getUrl } = useGetUrl();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const toaster = useAppToaster();

  const handleError = useCallback(() => {
    dispatch(clearInitialImage());
    if (shouldFetchImages) {
      toaster({
        title: 'Something went wrong, please refresh',
        status: 'error',
        isClosable: true,
      });
    } else {
      toaster({
        title: t('toast.parametersFailed'),
        description: t('toast.parametersFailedDesc'),
        status: 'error',
        isClosable: true,
      });
    }
  }, [dispatch, t, toaster, shouldFetchImages]);

  const handleChange = useCallback(
    (image: ImageDTO) => {
      dispatch(initialImageChanged(image));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(clearInitialImage());
  }, [dispatch]);

  return (
    <Flex
      sx={{
        width: 'full',
        height: 'full',
        position: 'relative',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <IAISelectableImage
        image={initialImage}
        onChange={handleChange}
        onReset={handleReset}
      />
    </Flex>
  );
};

export default InitialImagePreview;
