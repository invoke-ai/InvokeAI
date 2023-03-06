import { Image, useToast } from '@chakra-ui/react';
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
      <div className="init-image-preview-header">
        <h2>{t('parameters.initialImage')}</h2>
        <ImageUploaderIconButton />
      </div>
      {initialImage && (
        <div className="init-image-preview">
          <Image
            fit="contain"
            maxWidth="100%"
            maxHeight="100%"
            src={
              typeof initialImage === 'string' ? initialImage : initialImage.url
            }
            onError={alertMissingInitImage}
          />
        </div>
      )}
    </>
  );
}
