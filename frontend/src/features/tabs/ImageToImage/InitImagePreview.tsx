import { IconButton, Image, useToast } from '@chakra-ui/react';
import React, { SyntheticEvent } from 'react';
import { MdClear } from 'react-icons/md';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import { clearInitialImage } from '../../options/optionsSlice';

export default function InitImagePreview() {
  const { initialImage } = useAppSelector((state: RootState) => state.options);

  const dispatch = useAppDispatch();

  const toast = useToast();

  const handleClickResetInitialImage = (e: SyntheticEvent) => {
    e.stopPropagation();
    dispatch(clearInitialImage());
  };

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
        <h2>Initial Image</h2>
        <IconButton
          isDisabled={!initialImage}
          aria-label={'Reset Initial Image'}
          onClick={handleClickResetInitialImage}
          icon={<MdClear />}
        />
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
