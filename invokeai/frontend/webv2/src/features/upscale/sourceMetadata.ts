import { galleryImages } from '@features/gallery';
import { ApiError } from '@platform/transport/http';

/**
 * The metadata a Model only upscale copies into its result. Null when the image has none (an upload) or the server no
 * longer has it; any other failure throws, so the run is not queued without the metadata it should carry. Callers load
 * this module on demand, keeping its gallery read out of the editor's startup graph.
 */
export const readUpscaleSourceMetadata = async (
  imageName: string,
  signal?: AbortSignal
): Promise<Record<string, unknown> | null> => {
  try {
    return await galleryImages.metadata(imageName, signal);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null;
    }
    throw error;
  }
};
