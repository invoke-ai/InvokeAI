import type { GeneratedImageContract } from './types';

export const GALLERY_RECENT_IMAGE_LIMIT = 60;

const isGeneratedImageContract = (value: unknown): value is GeneratedImageContract => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const image = value as Partial<GeneratedImageContract>;

  return (
    typeof image.height === 'number' &&
    Number.isFinite(image.height) &&
    typeof image.imageName === 'string' &&
    image.imageName.length > 0 &&
    typeof image.imageUrl === 'string' &&
    typeof image.queuedAt === 'string' &&
    typeof image.sourceQueueItemId === 'string' &&
    typeof image.thumbnailUrl === 'string' &&
    typeof image.width === 'number' &&
    Number.isFinite(image.width)
  );
};

/**
 * Validates a newest-first recent-image overlay, retaining the newest occurrence
 * of each image name and bounding the persisted result.
 */
export const getBoundedRecentImages = (value: unknown): GeneratedImageContract[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const imageNames = new Set<string>();
  const images: GeneratedImageContract[] = [];

  for (const image of value) {
    if (!isGeneratedImageContract(image) || imageNames.has(image.imageName)) {
      continue;
    }

    imageNames.add(image.imageName);
    images.push(image);

    if (images.length === GALLERY_RECENT_IMAGE_LIMIT) {
      break;
    }
  }

  return images;
};
