import { Box, Flex, Image, Spinner, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import SelectImagePlaceholder from 'common/components/SelectImagePlaceholder';
import { useGetUrl } from 'common/util/getUrl';
import useGetImageByNameAndType from 'features/gallery/hooks/useGetImageByName';
import {
  clearInitialImage,
  initialImageSelected,
} from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash-es';
import { DragEvent, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ImageType } from 'services/api';
import ImageToImageOverlay from 'common/components/ImageToImageOverlay';
import { initialImageSelector } from 'features/parameters/store/generationSelectors';

const selector = createSelector(
  [initialImageSelector],
  (initialImage) => {
    return {
      initialImage,
    };
  },
  { memoizeOptions: { resultEqualityCheck: isEqual } }
);

const InitialImagePreview = () => {
  const isImageToImageEnabled = useAppSelector(
    (state: RootState) => state.generation.isImageToImageEnabled
  );
  const { initialImage } = useAppSelector(selector);
  const { getUrl } = useGetUrl();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const [isLoaded, setIsLoaded] = useState(false);
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
    setIsLoaded(false);
  };

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      setIsLoaded(false);
      const name = e.dataTransfer.getData('invokeai/imageName');
      const type = e.dataTransfer.getData('invokeai/imageType') as ImageType;

      if (!name || !type) {
        return;
      }

      const image = getImageByNameAndType(name, type);

      if (!image) {
        return;
      }

      dispatch(initialImageSelected({ name, type }));
    },
    [getImageByNameAndType, dispatch]
  );

  return (
    <Flex
      sx={{
        height: 'full',
        width: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
      }}
      onDrop={handleDrop}
    >
      {initialImage?.url && (
        <Box
          sx={{
            height: 'full',
            width: 'full',
            opacity: isImageToImageEnabled ? 1 : 0.5,
            filter: isImageToImageEnabled ? 'none' : 'auto',
            blur: '5px',
            position: 'relative',
          }}
        >
          <Image
            sx={{
              fit: 'contain',
              borderRadius: 'base',
            }}
            src={getUrl(initialImage?.url)}
            onError={onError}
            onLoad={() => {
              setIsLoaded(true);
            }}
            fallback={
              <Flex
                sx={{ h: 36, alignItems: 'center', justifyContent: 'center' }}
              >
                <Spinner color="grey" w="5rem" h="5rem" />
              </Flex>
            }
          />
          {isLoaded && <ImageToImageOverlay image={initialImage} />}
        </Box>
      )}
      {!initialImage?.url && <SelectImagePlaceholder />}
      {!isImageToImageEnabled && (
        <Flex
          sx={{
            w: 'full',
            h: 'full',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'absolute',
          }}
        >
          <Text
            fontWeight={500}
            fontSize="md"
            userSelect="none"
            color="base.200"
          >
            Image to Image is Disabled
          </Text>
        </Flex>
      )}
    </Flex>
  );
};

export default InitialImagePreview;
