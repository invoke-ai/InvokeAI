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
 * App-owned composition: uploads and own-account socket completions refresh the Gallery in visible tabs.
 * Only external origins whose batch asked for reveal can change Gallery selection.
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
      isVisible: () => document.visibilityState === 'visible',
      onVisibilityChange: (listener) => {
        document.addEventListener('visibilitychange', listener);
        return () => document.removeEventListener('visibilitychange', listener);
      },
    });

    runtime.start();

    return () => {
      runtime.dispose();
      revealer.dispose();
    };
  });

  return null;
};
