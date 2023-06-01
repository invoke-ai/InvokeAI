import { Flex, Image } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectImagesById } from 'features/gallery/store/imagesSlice';
import { DragEvent, memo, useCallback } from 'react';
import { ImageDTO } from 'services/api';

type ControlNetProcessorImageProps = {
  image: ImageDTO | undefined;
  setImage: (image: ImageDTO) => void;
};

const ControlNetProcessorImage = (props: ControlNetProcessorImageProps) => {
  const { image, setImage } = props;
  const state = useAppSelector((state) => state);
  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      const name = e.dataTransfer.getData('invokeai/imageName');
      const droppedImage = selectImagesById(state, name);
      if (droppedImage) {
        setImage(droppedImage);
      }
    },
    [setImage, state]
  );

  if (!image) {
    return <Flex onDrop={handleDrop}>Upload Image</Flex>;
  }

  return <Image src={image.image_url} />;
};

export default memo(ControlNetProcessorImage);
