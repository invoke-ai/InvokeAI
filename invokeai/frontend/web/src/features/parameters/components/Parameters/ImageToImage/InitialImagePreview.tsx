import { Flex, Image, Spinner } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import SelectImagePlaceholder from 'common/components/SelectImagePlaceholder';
import { useGetUrl } from 'common/util/getUrl';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { DragEvent, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ImageType } from 'services/api';
import ImageToImageOverlay from 'common/components/ImageToImageOverlay';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { initialImageSelected } from 'features/parameters/store/actions';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';

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
  const { getUrl } = useGetUrl();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const [isLoaded, setIsLoaded] = useState(false);

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

      dispatch(initialImageSelected({ name, type }));
    },
    [dispatch]
  );

  return (
    <Flex
      sx={{
        width: 'full',
        height: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
      }}
      onDrop={handleDrop}
    >
      <Flex
        sx={{
          height: 'full',
          width: 'full',
          blur: '5px',
          position: 'relative',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {initialImage?.url && (
          <>
            <Image
              sx={{
                objectFit: 'contain',
                borderRadius: 'base',
                maxHeight: 'full',
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
          </>
        )}
        {!initialImage?.url && <SelectImagePlaceholder />}
      </Flex>
    </Flex>
  );
};

export default InitialImagePreview;
