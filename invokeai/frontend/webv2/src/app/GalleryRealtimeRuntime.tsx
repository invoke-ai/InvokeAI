import { createGalleryRealtimeRuntime, invalidateGallery } from '@features/gallery/queries';
import { getSessionUserId } from '@features/identity';
import { isWebv2QueueItemOrigin } from '@features/queue/contracts';
import { useMountEffect } from '@platform/react/useMountEffect';
import { socketHub } from '@platform/transport/socketHub';
import { useQueryClient } from '@tanstack/react-query';
import { isEditableHotkeyTarget } from '@workbench/hotkeys/keys';
import { attachExternalGenerationReveal } from '@workbench/image-actions/recallParametersBridge';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';

/**
 * App-owned composition: uploads and other clients' completed generations the backend announces over the socket
 * refresh the Gallery read model, and those whose batch asked for it are revealed in the Gallery.
 */
export const GalleryRealtimeRuntime = () => {
  const queryClient = useQueryClient();
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();

  useMountEffect(() => {
    const revealer = attachExternalGenerationReveal({
      context: {
        commands,
        // A field keeps focus when the user switches to another window, such as the chatbot beside Invoke.
        isEditingText: () => document.hasFocus() && isEditableHotkeyTarget(document.activeElement),
        isVisible: () => document.visibilityState === 'visible',
        onVisibilityChange: (listener) => {
          document.addEventListener('visibilitychange', listener);
          return () => document.removeEventListener('visibilitychange', listener);
        },
        queries,
        queryClient,
      },
    });
    const runtime = createGalleryRealtimeRuntime({
      backend: socketHub,
      getSessionUserId,
      invalidate: () => invalidateGallery(queryClient),
      isClientQueueItemOrigin: isWebv2QueueItemOrigin,
      onRevealRequest: revealer.request,
    });

    runtime.start();

    return () => {
      runtime.dispose();
      revealer.dispose();
    };
  });

  return null;
};
