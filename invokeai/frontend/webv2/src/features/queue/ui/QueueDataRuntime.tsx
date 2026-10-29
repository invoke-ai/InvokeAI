import { refreshGenerationDevices } from '@features/queue/data/generationDevicesStore';
import { itemProgressStore } from '@features/queue/data/itemProgressStore';
import { invalidateQueueQueries } from '@features/queue/data/queries';
import { createProductionQueueRealtimeRuntime } from '@features/queue/publicApi';
import { queryClient } from '@platform/query/client';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useTranslation } from 'react-i18next';

import { refreshModelCacheStats } from './modelCacheStore';
import { takeQueueActionRequest, useQueueActionRequest, type QueueActionRequest } from './queueActionRequests';
import { clearQueueConfirmation, useQueueConfirmation } from './queueConfirmationStore';
import { useQueueLoadState } from './queueDataStore';
import { getQueueAction, useQueueActions } from './queueMenuActions';
import { useQueueUi } from './QueueUiContext';

export const attachQueueDataRuntime = (): (() => void) => {
  const runtime = createProductionQueueRealtimeRuntime({
    invalidate: () => invalidateQueueQueries(queryClient),
    progress: itemProgressStore,
    refreshModelCache: refreshModelCacheStats,
  });

  // Read device options once because changes require restart; hide labels for single-accelerator systems.
  void refreshGenerationDevices();

  runtime.start();

  return runtime.dispose;
};

const noop = () => undefined;

/**
 * Runs a requested action once, through the same model and confirmation as its menu item, after the queue's read model
 * has loaded: before that, every count reads zero and the action would look like it had nothing to do.
 */
const QueueActionRequestConsumer = ({ request }: { request: QueueActionRequest }) => {
  const { isConnected } = useQueueUi();
  const { loadState } = useQueueLoadState();

  return loadState === 'loading' && isConnected ? null : <RequestedQueueAction request={request} />;
};

const RequestedQueueAction = ({ request }: { request: QueueActionRequest }) => {
  const { t } = useTranslation();
  const { isConnected, notify } = useQueueUi();
  const action = getQueueAction(useQueueActions(), request.actionId);

  useMountEffect(() => {
    if (!takeQueueActionRequest(request.requestId) || !action) {
      return;
    }

    if (!action.disabled) {
      void action.trigger();
    } else {
      notify.info(
        action.label,
        isConnected ? t('widgets.queue.actionHasNothingToDo') : t('widgets.queue.unavailableWhileDisconnected')
      );
    }
  });

  return null;
};

/** React is only the idempotent lifecycle adapter for the non-React runtime. */
export const QueueDataRuntime = () => {
  const confirmation = useQueueConfirmation();
  const actionRequest = useQueueActionRequest();

  useMountEffect(attachQueueDataRuntime);

  // Always mounted, so a cleared confirmation animates out; the dialog keeps the text it showed while it closes.
  return (
    <>
      {actionRequest ? <QueueActionRequestConsumer key={actionRequest.requestId} request={actionRequest} /> : null}
      <ConfirmDialog
        body={confirmation?.body}
        cancelLabel={confirmation?.cancelLabel}
        confirmLabel={confirmation?.confirmLabel ?? ''}
        isOpen={confirmation !== null}
        title={confirmation?.title ?? ''}
        onClose={clearQueueConfirmation}
        onConfirm={confirmation?.onConfirm ?? noop}
      />
    </>
  );
};
