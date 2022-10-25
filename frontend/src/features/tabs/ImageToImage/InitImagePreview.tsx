import { IconButton, Image, useToast } from '@chakra-ui/react';
import React, { SyntheticEvent } from 'react';
import { MdClear } from 'react-icons/md';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import { setInitialImagePath } from '../../options/optionsSlice';

export default function InitImagePreview() {
  const initialImagePath = useAppSelector(
    (state: RootState) => state.options.initialImagePath
  );

  const dispatch = useAppDispatch();

  const toast = useToast();

  const handleClickResetInitialImage = (e: SyntheticEvent) => {
    e.stopPropagation();
    dispatch(setInitialImagePath(null));
  };

  const alertMissingInitImage = () => {
    toast({
      title: 'Problem loading parameters',
      description: 'Unable to load init image.',
      status: 'error',
      isClosable: true,
    });
    dispatch(setInitialImagePath(null));
  };

  return (
    <div className="init-image-preview">
      <div className="init-image-preview-header">
        <h1>Initial Image</h1>
        <IconButton
          isDisabled={!initialImagePath}
          size={'sm'}
          aria-label={'Reset Initial Image'}
          onClick={handleClickResetInitialImage}
          icon={<MdClear />}
        />
      </div>
      {initialImagePath && (
        <div className="init-image-image">
          <Image
            fit={'contain'}
            src={initialImagePath}
            rounded={'md'}
            onError={alertMissingInitImage}
          />
        </div>
      )}
    </div>
  );
}
