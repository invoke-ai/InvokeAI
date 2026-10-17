import { createGalleryRealtimeRuntime, invalidateGallery } from '@features/gallery/queries';
import { getSessionUserId } from '@features/identity';
import { isWebv2QueueItemOrigin } from '@features/queue/contracts';
import { useMountEffect } from '@platform/react/useMountEffect';
import { socketHub } from '@platform/transport/socketHub';
import { useQueryClient } from '@tanstack/react-query';

/**
 * App-owned composition: uploads and other clients' completed generations the backend announces over the socket
 * refresh the Gallery read model.
 */
export const GalleryRealtimeRuntime = () => {
  const queryClient = useQueryClient();

  useMountEffect(() => {
    const runtime = createGalleryRealtimeRuntime({
      backend: socketHub,
      getSessionUserId,
      invalidate: () => invalidateGallery(queryClient),
      isClientQueueItemOrigin: isWebv2QueueItemOrigin,
    });

    runtime.start();

    return runtime.dispose;
  });

  return null;
};
