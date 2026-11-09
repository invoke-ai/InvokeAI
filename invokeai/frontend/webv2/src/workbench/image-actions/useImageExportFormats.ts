import type { GalleryImage } from '@features/gallery';

import { galleryImageExportFormatsOptions } from '@features/gallery/queries';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { useQuery } from '@tanstack/react-query';

/**
 * Whether the server can export an image as a 16-bit PNG, asked when its menu mounts: hosts mount it on open, so
 * neither a grid of tiles nor a selection change requests anything. A null image (a bulk target) requests nothing.
 * Until the answer arrives, and after a failed request, it is false: the entry is absent rather than disabled, and
 * the next open asks again.
 */
export const useCanExportPng16 = (image: GalleryImage | null): boolean => {
  const query = useQuery({
    ...galleryImageExportFormatsOptions(captureAccountScope(), image?.imageName ?? ''),
    enabled: image !== null,
  });

  // A failed refetch keeps the previous data; it is no answer for this open.
  return !query.isError && query.data?.includes('png16') === true;
};
