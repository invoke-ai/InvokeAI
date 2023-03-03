import { Image } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';

export default function InitialImageOverlay() {
  const initialImage = useAppSelector(
    (state: RootState) => state.generation.initialImage
  );

  return initialImage ? (
    <Image
      fit="contain"
      src={typeof initialImage === 'string' ? initialImage : initialImage.url}
      rounded="md"
      className="checkerboard"
    />
  ) : null;
}
