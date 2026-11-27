import { deepClone } from 'common/util/deepClone';
import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasObjectState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  CanvasState,
  CroppableImageWithDims,
  ImageWithDims,
  RefImageState,
} from 'features/controlLayers/store/types';
import { getImageDTOSafe } from 'services/api/endpoints/images';
import { z } from 'zod';

export const CANVAS_PROJECT_VERSION = 1;
export const CANVAS_PROJECT_EXTENSION = '.invk';

// #region Manifest

const zCanvasProjectManifest = z.object({
  version: z.literal(CANVAS_PROJECT_VERSION),
  appVersion: z.string(),
  createdAt: z.string(),
  name: z.string(),
});
export type CanvasProjectManifest = z.infer<typeof zCanvasProjectManifest>;

export const parseManifest = (data: unknown): CanvasProjectManifest => {
  return zCanvasProjectManifest.parse(data);
};

// #endregion

// #region Canvas Project State

export type CanvasProjectState = {
  rasterLayers: CanvasRasterLayerState[];
  controlLayers: CanvasControlLayerState[];
  inpaintMasks: CanvasInpaintMaskState[];
  regionalGuidance: CanvasRegionalGuidanceState[];
  bbox: CanvasState['bbox'];
  selectedEntityIdentifier: CanvasState['selectedEntityIdentifier'];
  bookmarkedEntityIdentifier: CanvasState['bookmarkedEntityIdentifier'];
};

// #endregion

// #region Image Name Collection

/**
 * Collects image_name values from a CroppableImageWithDims (used by ref images).
 */
const collectFromCroppableImage = (image: CroppableImageWithDims | null, names: Set<string>): void => {
  if (!image) {
    return;
  }
  names.add(image.original.image.image_name);
  if (image.crop?.image) {
    names.add(image.crop.image.image_name);
  }
};

/**
 * Collects image_name values from an ImageWithDims (used by regional guidance ref images).
 */
const collectFromImageWithDims = (image: ImageWithDims | null, names: Set<string>): void => {
  if (!image) {
    return;
  }
  names.add(image.image_name);
};

/**
 * Collects image_name values from canvas objects (brush lines, images, etc.).
 */
const collectFromObjects = (objects: CanvasObjectState[], names: Set<string>): void => {
  for (const obj of objects) {
    if (obj.type === 'image' && 'image_name' in obj.image) {
      names.add(obj.image.image_name);
    }
  }
};

/**
 * Walks the entire canvas state + ref images and returns a deduplicated set of all image_name references.
 */
export const collectImageNames = (canvasState: CanvasProjectState, refImages: RefImageState[]): Set<string> => {
  const names = new Set<string>();

  // Raster layers
  for (const layer of canvasState.rasterLayers) {
    collectFromObjects(layer.objects, names);
  }

  // Control layers
  for (const layer of canvasState.controlLayers) {
    collectFromObjects(layer.objects, names);
  }

  // Inpaint masks
  for (const mask of canvasState.inpaintMasks) {
    collectFromObjects(mask.objects, names);
  }

  // Regional guidance
  for (const rg of canvasState.regionalGuidance) {
    collectFromObjects(rg.objects, names);
    for (const refImage of rg.referenceImages) {
      if (refImage.config.type === 'ip_adapter' || refImage.config.type === 'flux_redux') {
        collectFromImageWithDims(refImage.config.image, names);
      }
    }
  }

  // Global reference images
  for (const refImage of refImages) {
    collectFromCroppableImage(refImage.config.image, names);
  }

  return names;
};

// #endregion

// #region Image Name Remapping

/**
 * Remaps image_name values in a CroppableImageWithDims.
 */
/**
 * Remaps image_name values in a CroppableImageWithDims in-place.
 * Caller is responsible for cloning beforehand.
 */
const remapCroppableImage = (image: CroppableImageWithDims | null, mapping: Map<string, string>): void => {
  if (!image) {
    return;
  }

  const newOriginalName = mapping.get(image.original.image.image_name);
  if (newOriginalName) {
    image.original.image.image_name = newOriginalName;
  }

  if (image.crop?.image) {
    const newCropName = mapping.get(image.crop.image.image_name);
    if (newCropName) {
      image.crop.image.image_name = newCropName;
    }
  }
};

/**
 * Remaps image_name in an ImageWithDims.
 */
const remapImageWithDims = (image: ImageWithDims | null, mapping: Map<string, string>): ImageWithDims | null => {
  if (!image) {
    return null;
  }

  const result = deepClone(image);
  const newName = mapping.get(result.image_name);
  if (newName) {
    result.image_name = newName;
  }
  return result;
};

/**
 * Remaps image_name values in canvas objects.
 */
const remapObjects = (objects: CanvasObjectState[], mapping: Map<string, string>): CanvasObjectState[] => {
  return objects.map((obj) => {
    if (obj.type === 'image' && 'image_name' in obj.image) {
      const newName = mapping.get(obj.image.image_name);
      if (newName) {
        return { ...obj, image: { ...obj.image, image_name: newName } };
      }
    }
    return obj;
  });
};

/**
 * Deep-clones canvas state and remaps all image_name values using the provided mapping.
 * Only images present in the mapping are changed (images that already existed on the server are skipped).
 */
export const remapCanvasState = (canvasState: CanvasProjectState, mapping: Map<string, string>): CanvasProjectState => {
  if (mapping.size === 0) {
    return canvasState;
  }

  const result = deepClone(canvasState);

  for (const layer of result.rasterLayers) {
    layer.objects = remapObjects(layer.objects, mapping);
  }

  for (const layer of result.controlLayers) {
    layer.objects = remapObjects(layer.objects, mapping);
  }

  for (const mask of result.inpaintMasks) {
    mask.objects = remapObjects(mask.objects, mapping);
  }

  for (const rg of result.regionalGuidance) {
    rg.objects = remapObjects(rg.objects, mapping);
    for (const refImage of rg.referenceImages) {
      if (refImage.config.type === 'ip_adapter' || refImage.config.type === 'flux_redux') {
        refImage.config.image = remapImageWithDims(refImage.config.image, mapping);
      }
    }
  }

  return result;
};

/**
 * Deep-clones ref images and remaps all image_name values using the provided mapping.
 */
export const remapRefImages = (refImages: RefImageState[], mapping: Map<string, string>): RefImageState[] => {
  if (mapping.size === 0) {
    return refImages;
  }

  return refImages.map((refImage) => {
    const result = deepClone(refImage);
    remapCroppableImage(result.config.image, mapping);
    return result;
  });
};

// #endregion

// #region Concurrency

const MAX_CONCURRENT_REQUESTS = 5;

/**
 * Processes an array of async tasks with a concurrency limit.
 */
export const processWithConcurrencyLimit = async <T>(
  items: T[],
  fn: (item: T) => Promise<void>,
  limit: number = MAX_CONCURRENT_REQUESTS
): Promise<void> => {
  let index = 0;

  const next = async (): Promise<void> => {
    while (index < items.length) {
      const currentIndex = index++;
      await fn(items[currentIndex]!);
    }
  };

  const workers = Array.from({ length: Math.min(limit, items.length) }, () => next());
  await Promise.all(workers);
};

// #endregion

// #region Image Existence Check

/**
 * Checks which images already exist on the backend server.
 * Returns sets of existing and missing image names.
 */
export const checkExistingImages = async (
  imageNames: Set<string>
): Promise<{ existing: Set<string>; missing: Set<string> }> => {
  const existing = new Set<string>();
  const missing = new Set<string>();

  await processWithConcurrencyLimit(Array.from(imageNames), async (imageName) => {
    const dto = await getImageDTOSafe(imageName);
    if (dto) {
      existing.add(imageName);
    } else {
      missing.add(imageName);
    }
  });

  return { existing, missing };
};

// #endregion
