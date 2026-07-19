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

  runtime.start();

  return runtime.dispose;
};

/** React is only the idempotent lifecycle adapter for the non-React runtime. */
export const QueueDataRuntime = () => {
  const confirmation = useQueueConfirmation();

  useMountEffect(attachQueueDataRuntime);

  return confirmation ? (
    <ConfirmDialog
      body={confirmation.body}
      confirmLabel={confirmation.confirmLabel}
      isOpen={true}
      title={confirmation.title}
      onClose={clearQueueConfirmation}
      onConfirm={confirmation.onConfirm}
    />
  ) : null;
};
