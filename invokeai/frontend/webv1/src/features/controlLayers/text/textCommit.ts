import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import type { ImageDTO } from 'services/api/types';

export const getCommittedTextImageDimensions = (width: number, height: number) => ({
  width: Math.max(1, Math.ceil(width)),
  height: Math.max(1, Math.ceil(height)),
});

export const buildCommittedTextImageState = (imageDTO: ImageDTO, width: number, height: number) => {
  const committedDimensions = getCommittedTextImageDimensions(width, height);
  return imageDTOToImageObject(imageDTO, {
    image: {
      image_name: imageDTO.image_name,
      width: committedDimensions.width,
      height: committedDimensions.height,
    },
  });
};
