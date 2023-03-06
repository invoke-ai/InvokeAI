import { Flex, Image, Text, useToast } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import ImageUploaderIconButton from 'common/components/ImageUploaderIconButton';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function InitImagePreview() {
  const initialImage = useAppSelector(
    (state: RootState) => state.generation.initialImage
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const toast = useToast();

  const alertMissingInitImage = () => {
    toast({
      title: t('toast.parametersFailed'),
      description: t('toast.parametersFailedDesc'),
      status: 'error',
      isClosable: true,
    });
    dispatch(clearInitialImage());
  };

  return (
    <>
      <Flex
        sx={{
          alignItems: 'center',
          justifyContent: 'center',
          w: '100%',
          gap: 4,
        }}
      >
        <Text
          sx={{
            fontSize: 'lg',
          }}
          variant="subtext"
        >
          {t('parameters.initialImage')}
        </Text>
        <ImageUploaderIconButton />
      </Flex>
      {initialImage && (
        <Flex
          sx={{
            position: 'relative',
            height: '100%',
            width: '100%',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Image
            sx={{
              fit: 'contain',
              maxWidth: '100%',
              maxHeight: '100%',
              borderRadius: 'base',
              objectFit: 'contain',
              position: 'absolute',
            }}
            src={
              typeof initialImage === 'string' ? initialImage : initialImage.url
            }
            onError={alertMissingInitImage}
          />
        </Flex>
      )}
    </>
  );
}
