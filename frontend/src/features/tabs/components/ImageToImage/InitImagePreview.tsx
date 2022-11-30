import { Image, useToast } from '@chakra-ui/react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import ImageUploaderIconButton from 'common/components/ImageUploaderIconButton';
import { clearInitialImage } from 'features/options/store/optionsSlice';

export default function InitImagePreview() {
  const initialImage = useAppSelector(
    (state: RootState) => state.options.initialImage
  );

  const dispatch = useAppDispatch();

  const toast = useToast();

  // const handleClickResetInitialImage = (e: SyntheticEvent) => {
  //   e.stopPropagation();
  //   dispatch(clearInitialImage());
  // };

  const alertMissingInitImage = () => {
    toast({
      title: 'Problem loading parameters',
      description: 'Unable to load init image.',
      status: 'error',
      isClosable: true,
    });
    dispatch(clearInitialImage());
  };

  return (
    <>
      <div className="init-image-preview-header">
      {/* <div className="init-image-preview-header"> */}
        <h2>Initial Image</h2>
        {/* <IconButton
          isDisabled={!initialImage}
          aria-label={'Reset Initial Image'}
          onClick={handleClickResetInitialImage}
          icon={<MdClear />}
        /> */}
        <ImageUploaderIconButton />
      </div>
      {initialImage && (
        <div className="init-image-preview">
          <Image
            fit={'contain'}
            maxWidth={'100%'}
            maxHeight={'100%'}
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
