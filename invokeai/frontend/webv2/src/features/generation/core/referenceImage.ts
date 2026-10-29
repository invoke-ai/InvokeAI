import type { GeneratedImageContract } from '@features/gallery';

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

// ---- Crop-box geometry for the reference image crop editor -------------------------------------------------------
// Boxes are percentages of the original image, so an aspect ratio (width / height in pixels) becomes a width% per
// height% factor through the image's own proportions.

export type ReferenceCropAspectRatioId = 'free' | '16:9' | '3:2' | '4:3' | '1:1' | '3:4' | '2:3' | '9:16';

/** The legacy cropper's presets, in its order; `null` leaves the box free. */
export const REFERENCE_CROP_ASPECT_RATIOS: readonly { id: ReferenceCropAspectRatioId; ratio: number | null }[] = [
  { id: 'free', ratio: null },
  { id: '16:9', ratio: 16 / 9 },
  { id: '3:2', ratio: 3 / 2 },
  { id: '4:3', ratio: 4 / 3 },
  { id: '1:1', ratio: 1 },
  { id: '3:4', ratio: 3 / 4 },
  { id: '2:3', ratio: 2 / 3 },
  { id: '9:16', ratio: 9 / 16 },
];

export type ReferenceCropHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

export type ReferenceCropDragMode = 'move' | ReferenceCropHandle;

export interface ReferenceCropDrag {
  /** The box when the drag started. */
  box: ReferenceImageCropBoxPct;
  mode: ReferenceCropDragMode;
  /** The pointer position when the drag started, in image percent. */
  x: number;
  y: number;
}

export interface ReferenceCropConstraints {
  /** The original image size, which maps pixel ratios onto percent boxes. */
  image: { height: number; width: number };
  /** Minimum crop edge, in image pixels. */
  minDimensionPx: number;
  /** Pixel width / height to hold, or null for a free box. */
  ratio: number | null;
}

/** Tolerance when matching a box to a preset: a percent box rounded to pixels is never exact. */
const RATIO_MATCH_TOLERANCE = 0.005;

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const minSizePct = ({ image, minDimensionPx }: ReferenceCropConstraints): { height: number; width: number } => ({
  height: Math.min(100, (minDimensionPx / Math.max(1, image.height)) * 100),
  width: Math.min(100, (minDimensionPx / Math.max(1, image.width)) * 100),
});

/** Width percent per height percent for a pixel ratio on this image. */
const toPctRatio = (ratio: number, image: ReferenceCropConstraints['image']): number =>
  (ratio * Math.max(1, image.height)) / Math.max(1, image.width);

/** The preset whose pixel ratio the box has, or `free`. */
export const matchReferenceCropAspectRatio = (
  box: ReferenceImageCropBoxPct,
  image: ReferenceCropConstraints['image']
): ReferenceCropAspectRatioId => {
  const pixelRatio = (box.width * image.width) / Math.max(1e-6, box.height * image.height);
  const match = REFERENCE_CROP_ASPECT_RATIOS.find(
    ({ ratio }) => ratio !== null && Math.abs(pixelRatio / ratio - 1) <= RATIO_MATCH_TOLERANCE
  );
  return match?.id ?? 'free';
};

/**
 * A box of `pctRatio` whose width is at most `width`, at least the minimum edge, and that fits the image: the larger
 * of the minimum edges wins, and overflow shrinks the box rather than breaking the ratio.
 */
const fitRatioBox = (
  width: number,
  pctRatio: number,
  constraints: ReferenceCropConstraints
): { height: number; width: number } => {
  const min = minSizePct(constraints);
  let w = Math.max(width, min.width, min.height * pctRatio);
  w = Math.min(w, 100, 100 * pctRatio);
  return { height: w / pctRatio, width: w };
};

/** Re-centers the box on `(cx, cy)` and slides it back inside the image when the center sits near an edge. */
const placeAround = (cx: number, cy: number, size: { height: number; width: number }): ReferenceImageCropBoxPct => ({
  height: size.height,
  width: size.width,
  x: clamp(cx - size.width / 2, 0, 100 - size.width),
  y: clamp(cy - size.height / 2, 0, 100 - size.height),
});

/**
 * Applies a preset to an existing box: the area and center are preserved where the image allows, as the legacy
 * cropper did, so switching presets reshapes rather than relocates the crop. A null ratio returns the box as is.
 */
export const constrainReferenceCropBox = (
  box: ReferenceImageCropBoxPct,
  constraints: ReferenceCropConstraints
): ReferenceImageCropBoxPct => {
  if (constraints.ratio === null) {
    return box;
  }
  const pctRatio = toPctRatio(constraints.ratio, constraints.image);
  const size = fitRatioBox(Math.sqrt(box.width * box.height * pctRatio), pctRatio, constraints);
  return placeAround(box.x + box.width / 2, box.y + box.height / 2, size);
};

const resolveFreeDrag = (
  drag: ReferenceCropDrag,
  point: { x: number; y: number },
  constraints: ReferenceCropConstraints
): ReferenceImageCropBoxPct => {
  const dx = point.x - drag.x;
  const dy = point.y - drag.y;
  const { box, mode } = drag;
  const min = minSizePct(constraints);
  let left = box.x;
  let top = box.y;
  let right = box.x + box.width;
  let bottom = box.y + box.height;

  if (mode.includes('w')) {
    left = clamp(left + dx, 0, right - min.width);
  }
  if (mode.includes('e')) {
    right = clamp(right + dx, left + min.width, 100);
  }
  if (mode.includes('n')) {
    top = clamp(top + dy, 0, bottom - min.height);
  }
  if (mode.includes('s')) {
    bottom = clamp(bottom + dy, top + min.height, 100);
  }
  return { height: bottom - top, width: right - left, x: left, y: top };
};

/**
 * A ratio-locked handle drag. Corners anchor the opposite corner and follow the larger ratio-normalized displacement;
 * edges anchor the opposite edge, centering the other dimension until an image edge requires it to slide.
 */
const resolveRatioDrag = (
  drag: ReferenceCropDrag,
  point: { x: number; y: number },
  constraints: ReferenceCropConstraints,
  ratio: number
): ReferenceImageCropBoxPct => {
  const { box, mode } = drag;
  const pctRatio = toPctRatio(ratio, constraints.image);
  const free = resolveFreeDrag(drag, point, constraints);
  const anchorLeft = mode.includes('w') ? box.x + box.width : box.x;
  const anchorTop = mode.includes('n') ? box.y + box.height : box.y;
  const horizontal = mode === 'e' || mode === 'w';
  const vertical = mode === 'n' || mode === 's';

  // The room between the anchor and the image edge on the growing side.
  const roomWidth = mode.includes('w') ? anchorLeft : 100 - anchorLeft;
  const roomHeight = mode.includes('n') ? anchorTop : 100 - anchorTop;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;

  let width: number;
  if (vertical) {
    width = free.height * pctRatio;
  } else if (horizontal) {
    width = free.width;
  } else {
    width =
      Math.abs(free.width - box.width) >= Math.abs(free.height - box.height) * pctRatio
        ? free.width
        : free.height * pctRatio;
  }
  let size = fitRatioBox(width, pctRatio, constraints);
  // Side handles may use the full orthogonal image dimension; placement below slides their center if needed.
  const maxWidth = horizontal
    ? Math.min(roomWidth, 100 * pctRatio)
    : vertical
      ? Math.min(roomHeight * pctRatio, 100)
      : Math.min(roomWidth, roomHeight * pctRatio);
  if (size.width > maxWidth) {
    size = fitRatioBox(maxWidth, pctRatio, constraints);
  }

  const x = horizontal || !vertical ? (mode.includes('w') ? anchorLeft - size.width : anchorLeft) : cx - size.width / 2;
  const y = vertical || !horizontal ? (mode.includes('n') ? anchorTop - size.height : anchorTop) : cy - size.height / 2;
  return {
    height: size.height,
    width: size.width,
    x: clamp(x, 0, 100 - size.width),
    y: clamp(y, 0, 100 - size.height),
  };
};

/** The box after moving the pointer to `point` during `drag`. */
export const resolveReferenceCropDrag = (
  drag: ReferenceCropDrag,
  point: { x: number; y: number },
  constraints: ReferenceCropConstraints
): ReferenceImageCropBoxPct => {
  if (drag.mode === 'move') {
    const { box } = drag;
    return {
      ...box,
      x: clamp(box.x + point.x - drag.x, 0, 100 - box.width),
      y: clamp(box.y + point.y - drag.y, 0, 100 - box.height),
    };
  }
  return constraints.ratio === null
    ? resolveFreeDrag(drag, point, constraints)
    : resolveRatioDrag(drag, point, constraints, constraints.ratio);
};
