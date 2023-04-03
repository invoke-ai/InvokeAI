import { ResultImage } from 'app/invokeai';
import { ImageField, ImageType } from 'services/api';

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

export const extractTimestampFromResultImageName = (imageName: string) => {
  const timestamp = imageName.split('_')?.pop()?.split('.')[0];

  if (timestamp === undefined) {
    return 0;
  }

  return Number(timestamp);
};

export const prepareResultImage = (image: ImageField): ResultImage => {
  const name = image.image_name;

  const { imageUrl, thumbnailUrl } = buildImageUrls('results', name);

  const timestamp = extractTimestampFromResultImageName(name);

  return {
    name,
    url: imageUrl,
    thumbnail: thumbnailUrl,
    timestamp,
    height: 512,
    width: 512,
  };
};
