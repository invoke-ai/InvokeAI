import { Flex, Image, Text, useToast } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import ImageUploaderIconButton from 'common/components/ImageUploaderIconButton';
import { useGetUrl } from 'common/util/getUrl';
import { initialImageSelector } from 'features/parameters/store/generationSelectors';
import CurrentImageHidden from 'features/gallery/components/CurrentImageHidden';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';
import { RootState } from 'app/store';

export default function InitImagePreview() {
  const initialImage = useAppSelector(initialImageSelector);

  const { shouldHidePreview } = useAppSelector((state: RootState) => state.ui);

  const { t } = useTranslation();

  const dispatch = useAppDispatch();
  const { getUrl } = useGetUrl();

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
              shouldHidePreview
                ? undefined
                : typeof initialImage === 'string'
                ? getUrl(initialImage)
                : getUrl(initialImage.url)
            }
            fallback={<CurrentImageHidden />}
            onError={alertMissingInitImage}
          />
        </Flex>
      )}
    </>
  );
}
