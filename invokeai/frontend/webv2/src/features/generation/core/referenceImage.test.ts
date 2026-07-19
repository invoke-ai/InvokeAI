import { describe, expect, it } from 'vitest';

import {
  applyReferenceImageCrop,
  clearReferenceImageCrop,
  cloneCroppableImage,
  FULL_REFERENCE_IMAGE_CROP_BOX,
  generatedImageToReferenceImage,
  getEffectiveReferenceImage,
  getReferenceImageCropBoxPct,
  isCanonicalCroppableImage,
  normalizeCroppableImage,
  resolveReferenceImageCrop,
} from './referenceImage';

const original = { height: 768, image_name: 'original.png', width: 1024 };
const crop = {
  box: { height: 400, width: 600, x: 100, y: 80 },
  image: { height: 400, image_name: 'crop.png', width: 600 },
  ratio: null,
};

describe('reference images', () => {
  it('normalizes canonical uncropped and cropped assets', () => {
    expect(normalizeCroppableImage({ original: { image: original } })).toEqual({ original: { image: original } });
    expect(normalizeCroppableImage({ crop, original: { image: original } })).toEqual({
      crop,
      original: { image: original },
    });
  });

  it('migrates pre-crop legacy and former flat webv2 assets', () => {
    expect(isCanonicalCroppableImage(original)).toBe(false);
    expect(normalizeCroppableImage(original)).toEqual({ original: { image: original } });
    expect(
      normalizeCroppableImage({
        height: 512,
        imageName: 'webv2.png',
        imageUrl: '/full',
        queuedAt: 'now',
        sourceQueueItemId: 'queue',
        thumbnailUrl: '/thumbnail',
        width: 640,
      })
    ).toEqual({ original: { image: { height: 512, image_name: 'webv2.png', width: 640 } } });
    expect(isCanonicalCroppableImage({ original: { image: original } })).toBe(true);
    expect(
      isCanonicalCroppableImage({
        original: { image: { height: 768, imageName: 'not-canonical.png', width: 1024 } },
      })
    ).toBe(false);
  });

  it('converts gallery results to uncropped canonical assets', () => {
    expect(generatedImageToReferenceImage({ height: 300, imageName: 'upload.png', width: 400 })).toEqual({
      original: { image: { height: 300, image_name: 'upload.png', width: 400 } },
    });
  });

  it('uses a crop when present and preserves nested provenance when cloning', () => {
    const asset = { crop, original: { image: original } };
    const cloned = cloneCroppableImage(asset);

    expect(getEffectiveReferenceImage(asset)).toEqual(crop.image);
    expect(cloned).toEqual(asset);
    expect(cloned).not.toBe(asset);
    expect(cloned.original).not.toBe(asset.original);
    expect(cloned.crop).not.toBe(asset.crop);
    expect(cloned.crop?.box).not.toBe(asset.crop.box);
  });

  it('applies, replaces, and clears crops without replacing the original', () => {
    const uncropped = { original: { image: original } };
    const cropped = applyReferenceImageCrop(uncropped, crop);
    const replacement = applyReferenceImageCrop(cropped, {
      box: { height: 256, width: 256, x: 0, y: 0 },
      image: { height: 256, image_name: 'replacement.png', width: 256 },
      ratio: 1,
    });

    expect(cropped.original.image).toEqual(original);
    expect(replacement.original.image).toEqual(original);
    expect(getEffectiveReferenceImage(replacement).image_name).toBe('replacement.png');
    expect(clearReferenceImageCrop(replacement)).toEqual(uncropped);
  });

  it('initializes the crop editor from the saved pixel box', () => {
    expect(getReferenceImageCropBoxPct({ crop, original: { image: original } })).toEqual({
      height: (400 / 768) * 100,
      width: (600 / 1024) * 100,
      x: (100 / 1024) * 100,
      y: (80 / 768) * 100,
    });
  });

  it('resolves a full-image edit by clearing the crop', () => {
    expect(
      resolveReferenceImageCrop({ crop, original: { image: original } }, FULL_REFERENCE_IMAGE_CROP_BOX, null)
    ).toEqual({ original: { image: original } });
  });

  it('records an uploaded crop against the unchanged original', () => {
    const uploadedCrop = { height: 384, image_name: 'uploaded-crop.png', width: 512 };
    const result = resolveReferenceImageCrop(
      { crop, original: { image: original } },
      { height: 50, width: 50, x: 25, y: 10 },
      uploadedCrop
    );

    expect(result).toEqual({
      crop: {
        box: { height: 384, width: 512, x: 256, y: 77 },
        image: uploadedCrop,
        ratio: null,
      },
      original: { image: original },
    });
    expect(getEffectiveReferenceImage(result)).toEqual(uploadedCrop);
  });

  it('rejects malformed originals and crops', () => {
    expect(
      normalizeCroppableImage({ original: { image: { image_name: 'bad.png', width: 0, height: 10 } } })
    ).toBeNull();
    expect(
      normalizeCroppableImage({
        crop: { ...crop, box: { ...crop.box, x: -1 } },
        original: { image: original },
      })
    ).toBeNull();
  });
});
