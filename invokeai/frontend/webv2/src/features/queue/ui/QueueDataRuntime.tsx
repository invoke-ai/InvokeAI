import { refreshGenerationDevices } from '@features/queue/data/generationDevicesStore';
import { itemProgressStore } from '@features/queue/data/itemProgressStore';
import { createProductionQueueRealtimeRuntime } from '@features/queue/publicApi';
import { queryClient } from '@platform/query/client';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';

import { refreshModelCacheStats } from './modelCacheStore';
import { clearQueueConfirmation, useQueueConfirmation } from './queueConfirmationStore';

export const attachQueueDataRuntime = (): (() => void) => {
  const runtime = createProductionQueueRealtimeRuntime({
    invalidate: () => queryClient.invalidateQueries({ queryKey: ['queue'] }),
    progress: itemProgressStore,
    refreshModelCache: refreshModelCacheStats,
  });

  // Read device options once because changes require restart; hide labels for single-accelerator systems.
  void refreshGenerationDevices();

  runtime.start();

  return runtime.dispose;
};

const noop = () => undefined;

/** React is only the idempotent lifecycle adapter for the non-React runtime. */
export const QueueDataRuntime = () => {
  const confirmation = useQueueConfirmation();

  useMountEffect(attachQueueDataRuntime);

  // Always mounted, so a cleared confirmation animates out; the dialog keeps the text it showed while it closes.
  return (
    <ConfirmDialog
      body={confirmation?.body}
      confirmLabel={confirmation?.confirmLabel ?? ''}
      isOpen={confirmation !== null}
      title={confirmation?.title ?? ''}
      onClose={clearQueueConfirmation}
      onConfirm={confirmation?.onConfirm ?? noop}
    />
  );
};
