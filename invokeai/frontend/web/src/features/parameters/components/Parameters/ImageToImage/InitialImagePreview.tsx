import { Flex, Icon, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGetUrl } from 'common/util/getUrl';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { DragEvent, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { initialImageSelected } from 'features/parameters/store/actions';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import ImageFallbackSpinner from 'features/gallery/components/ImageFallbackSpinner';
import { FaImage } from 'react-icons/fa';
import { configSelector } from '../../../../system/store/configSelectors';
import { useAppToaster } from 'app/components/Toaster';

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

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      const name = e.dataTransfer.getData('invokeai/imageName');
      dispatch(initialImageSelected(name));
    },
    [dispatch]
  );

  return (
    <Flex
      sx={{
        width: 'full',
        height: 'full',
        position: 'relative',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onDrop={handleDrop}
    >
      {initialImage?.image_url && (
        <>
          <Image
            src={getUrl(initialImage?.image_url)}
            fallbackStrategy="beforeLoadOrError"
            fallback={<ImageFallbackSpinner />}
            onError={handleError}
            sx={{
              objectFit: 'contain',
              maxWidth: '100%',
              maxHeight: '100%',
              height: 'auto',
              position: 'absolute',
              borderRadius: 'base',
            }}
          />
          <ImageMetadataOverlay image={initialImage} />
        </>
      )}
      {!initialImage?.image_url && (
        <Icon
          as={FaImage}
          sx={{
            boxSize: 24,
            color: 'base.500',
          }}
        />
      )}
    </Flex>
  );
};

export default InitialImagePreview;
