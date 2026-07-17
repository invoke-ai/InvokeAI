import type { GeneratedImageContract } from '@workbench/types';

import { getImageFullUrl, getImageThumbnailUrl } from '@workbench/gallery/api';

import type { CroppableImageWithDims, ImageWithDims } from './types';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const isPositiveFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

export const isImageWithDims = (value: unknown): value is ImageWithDims =>
  isRecord(value) &&
  typeof value.image_name === 'string' &&
  value.image_name.length > 0 &&
  isPositiveFiniteNumber(value.width) &&
  isPositiveFiniteNumber(value.height);

const cloneImageWithDims = (image: ImageWithDims): ImageWithDims => ({ ...image });

const normalizeFlatImage = (value: unknown): ImageWithDims | null => {
  if (isImageWithDims(value)) {
    return cloneImageWithDims(value);
  }

  if (
    isRecord(value) &&
    typeof value.imageName === 'string' &&
    value.imageName.length > 0 &&
    isPositiveFiniteNumber(value.width) &&
    isPositiveFiniteNumber(value.height)
  ) {
    return { height: value.height, image_name: value.imageName, width: value.width };
  }

  return null;
};

const normalizeCrop = (value: unknown): CroppableImageWithDims['crop'] | undefined => {
  if (!isRecord(value) || !isRecord(value.box)) {
    return undefined;
  }

  const image = normalizeFlatImage(value.image);
  const { box } = value;
  const ratio = value.ratio;

  if (
    !image ||
    !isPositiveFiniteNumber(box.width) ||
    !isPositiveFiniteNumber(box.height) ||
    typeof box.x !== 'number' ||
    !Number.isFinite(box.x) ||
    box.x < 0 ||
    typeof box.y !== 'number' ||
    !Number.isFinite(box.y) ||
    box.y < 0 ||
    (ratio !== null && !isPositiveFiniteNumber(ratio))
  ) {
    return undefined;
  }

  return {
    box: { height: box.height, width: box.width, x: box.x, y: box.y },
    image,
    ratio: ratio as number | null,
  };
};

/** Converts a gallery/upload response into a canonical, uncropped reference asset. */
export const generatedImageToReferenceImage = (
  image: Pick<GeneratedImageContract, 'height' | 'imageName' | 'width'>
): CroppableImageWithDims => ({
  original: { image: { height: image.height, image_name: image.imageName, width: image.width } },
});

/**
 * Accepts canonical legacy data, pre-crop `{ image_name, width, height }`, and
 * webv2's former flat `GeneratedImageContract`. Flat webv2 crops are necessarily
 * migrated as a new original because their previous original was not retained.
 */
export const normalizeCroppableImage = (value: unknown): CroppableImageWithDims | null => {
  const flat = normalizeFlatImage(value);
  if (flat) {
    return { original: { image: flat } };
  }

  if (!isRecord(value) || !isRecord(value.original)) {
    return null;
  }

  const original = normalizeFlatImage(value.original.image);
  if (!original) {
    return null;
  }

  const crop = normalizeCrop(value.crop);
  if (value.crop !== undefined && !crop) {
    return null;
  }

  return { ...(crop ? { crop } : {}), original: { image: original } };
};

/** True only for the canonical nested representation, not accepted migration inputs. */
export const isCanonicalCroppableImage = (value: unknown): value is CroppableImageWithDims =>
  isRecord(value) &&
  isRecord(value.original) &&
  isImageWithDims(value.original.image) &&
  (value.crop === undefined ||
    (isRecord(value.crop) && isImageWithDims(value.crop.image) && normalizeCrop(value.crop) !== undefined));

export type ReferenceImageCropBoxPct = { height: number; width: number; x: number; y: number };

export const FULL_REFERENCE_IMAGE_CROP_BOX: ReferenceImageCropBoxPct = {
  height: 100,
  width: 100,
  x: 0,
  y: 0,
};

export const isFullReferenceImageCropBox = (cropBox: ReferenceImageCropBoxPct): boolean =>
  cropBox.x === 0 && cropBox.y === 0 && cropBox.width === 100 && cropBox.height === 100;

export const getReferenceImageCropBoxPct = (image: CroppableImageWithDims): ReferenceImageCropBoxPct => {
  if (!image.crop) {
    return FULL_REFERENCE_IMAGE_CROP_BOX;
  }

  return {
    height: (image.crop.box.height / image.original.image.height) * 100,
    width: (image.crop.box.width / image.original.image.width) * 100,
    x: (image.crop.box.x / image.original.image.width) * 100,
    y: (image.crop.box.y / image.original.image.height) * 100,
  };
};

export const resolveReferenceImageCrop = (
  image: CroppableImageWithDims,
  cropBox: ReferenceImageCropBoxPct,
  uploadedCrop: ImageWithDims | null
): CroppableImageWithDims => {
  if (isFullReferenceImageCropBox(cropBox)) {
    return clearReferenceImageCrop(image);
  }

  if (!uploadedCrop) {
    throw new Error('A cropped image is required for a partial crop.');
  }

  const original = image.original.image;
  return applyReferenceImageCrop(image, {
    box: {
      height: Math.max(1, Math.round((cropBox.height / 100) * original.height)),
      width: Math.max(1, Math.round((cropBox.width / 100) * original.width)),
      x: Math.round((cropBox.x / 100) * original.width),
      y: Math.round((cropBox.y / 100) * original.height),
    },
    image: uploadedCrop,
    ratio: null,
  });
};

export const cloneCroppableImage = (image: CroppableImageWithDims): CroppableImageWithDims => ({
  ...(image.crop
    ? {
        crop: {
          box: { ...image.crop.box },
          image: cloneImageWithDims(image.crop.image),
          ratio: image.crop.ratio,
        },
      }
    : {}),
  original: { image: cloneImageWithDims(image.original.image) },
});

export const getEffectiveReferenceImage = (image: CroppableImageWithDims): ImageWithDims =>
  image.crop?.image ?? image.original.image;

export const applyReferenceImageCrop = (
  image: CroppableImageWithDims,
  crop: NonNullable<CroppableImageWithDims['crop']>
): CroppableImageWithDims => ({
  crop: {
    box: { ...crop.box },
    image: cloneImageWithDims(crop.image),
    ratio: crop.ratio,
  },
  original: { image: cloneImageWithDims(image.original.image) },
});

export const clearReferenceImageCrop = (image: CroppableImageWithDims): CroppableImageWithDims => ({
  original: { image: cloneImageWithDims(image.original.image) },
});

export const getReferenceImageUrls = (image: CroppableImageWithDims) => {
  const { image_name } = getEffectiveReferenceImage(image);
  return { imageUrl: getImageFullUrl(image_name), thumbnailUrl: getImageThumbnailUrl(image_name) };
};
