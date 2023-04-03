import { ImageType } from 'services/api';

export const buildImageUrls = (
  imageType: ImageType,
  imageName: string
): { imageUrl: string; thumbnailUrl: string } => {
  const imageUrl = `api/v1/images/${imageType}/${imageName}`;

  const thumbnailUrl = `api/v1/images/${imageType}/thumbnails/${
    imageName.split('.')[0]
  }.webp`;

  return {
    imageUrl,
    thumbnailUrl,
  };
};
