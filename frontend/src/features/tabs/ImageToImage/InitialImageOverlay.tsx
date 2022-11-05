import { Image } from '@chakra-ui/react';
import React from 'react';
import { RootState, useAppSelector } from '../../../app/store';

export default function InitialImageOverlay() {
  const initialImage = useAppSelector(
    (state: RootState) => state.options.initialImage
  );

  return initialImage ? (
    <Image
      fit={'contain'}
      src={typeof initialImage === 'string' ? initialImage : initialImage.url}
      rounded={'md'}
      className={'checkerboard'}
    />
  ) : null;
}
