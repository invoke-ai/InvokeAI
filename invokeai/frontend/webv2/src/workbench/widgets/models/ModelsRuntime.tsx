import { useEffect, useRef } from 'react';

import { refreshInstalls, useInstallOutcomes } from '../../models/installsStore';
import { refreshModels } from '../../models/modelsStore';
import { useNotify } from '../../useNotify';
import { useWorkbench } from '../../WorkbenchContext';

/**
 * Renders nothing. Keeps the model stores honest across the app lifecycle:
 * refreshes them when the backend (re)connects, and turns install outcomes
 * (completed/failed/cancelled) into notifications even when no model manager
 * surface is mounted — a download that finishes 20 minutes later still toasts.
 */
export const ModelsRuntime = () => {
  const notify = useNotify();
  const { state } = useWorkbench();
  const outcomes = useInstallOutcomes();
  const seenOutcomeIdsRef = useRef<Set<number> | null>(null);
  const connectionStatus = state.backendConnection.status;

  useEffect(() => {
    if (connectionStatus === 'connected') {
      // Reconnects can miss socket events: re-sync both stores.
      void refreshModels();
      void refreshInstalls();
    }
  }, [connectionStatus]);

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

  return null;
};
