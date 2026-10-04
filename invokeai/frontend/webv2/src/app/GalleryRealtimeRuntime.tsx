import { createGalleryRealtimeRuntime, invalidateGallery } from '@features/gallery/queries';
import { useMountEffect } from '@platform/react/useMountEffect';
import { socketHub } from '@platform/transport/socketHub';
import { useQueryClient } from '@tanstack/react-query';

/** App-owned composition: uploads the backend announces over the socket refresh the Gallery read model. */
export const GalleryRealtimeRuntime = () => {
  const queryClient = useQueryClient();

  useMountEffect(() => {
    const runtime = createGalleryRealtimeRuntime({
      backend: socketHub,
      invalidate: () => invalidateGallery(queryClient),
    });

    runtime.start();

    return runtime.dispose;
  });

  return null;
};
