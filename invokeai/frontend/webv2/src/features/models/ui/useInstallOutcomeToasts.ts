import { useInstallOutcomes } from '@features/models/data/installsStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useEffect, useRef } from 'react';

/**
 * Turns settled install outcomes (completed/failed/cancelled) into notifications,
 * while the model manager is mounted. Provider-free: `useNotify` falls back to
 * global toasts off-workbench. Mounted by `ModelInstallRuntime`.
 */
export const useInstallOutcomeToasts = (): void => {
  const notify = useNotify();
  const outcomes = useInstallOutcomes();
  const seenOutcomeIdsRef = useRef<Set<number> | null>(null);

  useEffect(() => {
    if (seenOutcomeIdsRef.current === null) {
      seenOutcomeIdsRef.current = new Set(outcomes.map((outcome) => outcome.id));

      return;
    }

    for (const outcome of [...outcomes].reverse()) {
      if (seenOutcomeIdsRef.current.has(outcome.id)) {
        continue;
      }

      seenOutcomeIdsRef.current.add(outcome.id);

      if (outcome.kind === 'completed') {
        notify.success('Model installed', outcome.modelName ?? outcome.source);
      } else if (outcome.kind === 'error') {
        notify.error('Model install failed', `${outcome.source}: ${outcome.error ?? 'Unknown error.'}`);
      } else {
        notify.info('Model install cancelled', outcome.source);
      }
    }
  }, [notify, outcomes]);
};
