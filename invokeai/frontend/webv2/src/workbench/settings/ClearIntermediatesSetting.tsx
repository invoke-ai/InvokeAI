import { HStack, Stack, Text } from '@chakra-ui/react';
import { useCapabilities } from '@workbench/auth/capabilities';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button, ConfirmDialog } from '@workbench/components/ui';
import { createExternalStore } from '@workbench/externalStore';
import {
  clearIntermediates,
  getClearIntermediatesConfirmation,
  getClearIntermediatesState,
  getIntermediatesCount,
} from '@workbench/images/intermediates';
import { useNotify } from '@workbench/useNotify';
import { useQueueCounts } from '@workbench/widgets/queue/queueDataStore';
import { getQueueStatus } from '@workbench/widgets/queue/queueServerApi';
import { useOptionalWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { Trash2Icon } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

interface ClearIntermediatesSnapshot {
  error: string | null;
  hasGlobalQueueWork: boolean;
  intermediatesCount: number | null;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
}

const clearIntermediatesStore = createExternalStore<ClearIntermediatesSnapshot>({
  error: null,
  hasGlobalQueueWork: false,
  intermediatesCount: null,
  loadState: 'idle',
});

let inflight: Promise<void> | null = null;

const hasQueueWork = ({ queue }: Awaited<ReturnType<typeof getQueueStatus>>): boolean =>
  queue.pending > 0 || queue.in_progress > 0;

const refreshClearIntermediatesState = (): Promise<void> => {
  if (inflight) {
    return inflight;
  }

  clearIntermediatesStore.patchSnapshot({ error: null, loadState: 'loading' });

  inflight = Promise.all([getIntermediatesCount(), getQueueStatus()])
    .then(([count, queueStatus]) => {
      clearIntermediatesStore.patchSnapshot({
        error: null,
        hasGlobalQueueWork: hasQueueWork(queueStatus),
        intermediatesCount: count,
        loadState: 'loaded',
      });
    })
    .catch((error: unknown) => {
      clearIntermediatesStore.patchSnapshot({
        error: getApiErrorMessage(error, 'Could not load intermediate images.'),
        loadState: 'error',
      });
    })
    .finally(() => {
      inflight = null;
    });

  return inflight;
};

export const ClearIntermediatesSetting = () => {
  const { canClearIntermediates } = useCapabilities();
  const dispatch = useOptionalWorkbenchDispatch();
  const notify = useNotify();
  const queueCounts = useQueueCounts();
  const { error, hasGlobalQueueWork, intermediatesCount, loadState } = clearIntermediatesStore.useSnapshot();
  const [busy, setBusy] = useState(false);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);

  useEffect(() => {
    void refreshClearIntermediatesState();
  }, []);

  const state = getClearIntermediatesState({
    canClearIntermediates,
    hasActiveQueueWork: hasGlobalQueueWork || queueCounts.pending > 0 || queueCounts.in_progress > 0,
    intermediatesCount,
  });
  const disabledReason = loadState === 'error' ? error : state.reason;
  const label = intermediatesCount === null ? 'Clear intermediates' : `Clear intermediates (${intermediatesCount})`;
  const confirmation = getClearIntermediatesConfirmation(intermediatesCount ?? 0);

  const openConfirm = useCallback(() => {
    if (!state.disabled) {
      setIsConfirmOpen(true);
    }
  }, [state.disabled]);

  const closeConfirm = useCallback(() => setIsConfirmOpen(false), []);

  const handleClear = useCallback(async () => {
    if (state.disabled || busy) {
      return;
    }

    setBusy(true);

    try {
      const queueStatus = await getQueueStatus();
      const queueHasWork = hasQueueWork(queueStatus);
      clearIntermediatesStore.patchSnapshot({ hasGlobalQueueWork: queueHasWork });

      if (queueHasWork) {
        notify.info('Intermediates not cleared', 'Wait for pending or running queue work to finish.');
        return;
      }

      const clearedCount = await clearIntermediates();
      clearIntermediatesStore.patchSnapshot({ intermediatesCount: 0, loadState: 'loaded' });
      dispatch?.({ type: 'clearCanvasImageReferences' });
      dispatch?.({ type: 'refreshBackendData' });
      notify.info(
        'Intermediates cleared',
        `Cleared ${clearedCount} intermediate image${clearedCount === 1 ? '' : 's'}.`
      );
    } catch (error) {
      notify.error('Failed to clear intermediates', getApiErrorMessage(error, 'Could not clear intermediate images.'));
    } finally {
      setBusy(false);
    }
  }, [busy, dispatch, notify, state.disabled]);

  return (
    <Stack gap="2">
      <HStack alignItems="center" gap="3" justifyContent="space-between">
        <Stack gap="0.5">
          <Text color="fg" fontSize="sm" fontWeight="500">
            Intermediate images
          </Text>
          <Text color="fg.subtle" fontSize="xs">
            Clear temporary workflow outputs and progress artifacts. Final gallery images are not removed.
          </Text>
        </Stack>
        <Button
          disabled={state.disabled || loadState === 'error'}
          loading={busy || loadState === 'idle' || loadState === 'loading'}
          size="sm"
          variant="outline"
          onClick={openConfirm}
        >
          <Trash2Icon />
          {label}
        </Button>
      </HStack>
      {disabledReason ? (
        <Text color="fg.subtle" fontSize="2xs">
          {disabledReason}
        </Text>
      ) : null}
      <ConfirmDialog
        body={confirmation.body}
        confirmLabel={confirmation.confirmLabel}
        isOpen={isConfirmOpen}
        title={confirmation.title}
        onClose={closeConfirm}
        onConfirm={handleClear}
      />
    </Stack>
  );
};
