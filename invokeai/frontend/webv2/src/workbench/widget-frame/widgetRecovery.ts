import type { RegisteredWidget } from '@workbench/widgetContracts';

import { createLogger } from '@platform/logging/logger';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
import { describeWorkbenchReloadError, WorkbenchReloadError } from '@workbench/projects/reloadWorkbench';
import { useActiveProjectSelector, useWorkbenchInternalStore, useWorkbenchReload } from '@workbench/WorkbenchContext';
import { useCallback, useRef, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

export const useWidgetRecovery = (
  instanceId: string,
  widget: RegisteredWidget,
  slot = 'compact',
  region = 'bottom'
) => {
  const owner = useWorkbenchInternalStore();
  const account = captureAccountScope();
  const projectId = useActiveProjectSelector((project) => project.id);
  const recovery = owner.widgetRecovery.get(instanceId)!;
  const prefix = `${region}:`;
  const slotKey = prefix + slot;
  const key = (name: string) => prefix + name;
  // The registry resource can outlive this view; a reload requested here cannot.
  const lease = useRef({ active: true, generation: 0 });
  const subscribe = useCallback(
    (listener: () => void) => {
      lease.current.active = true;
      const unsubscribe = recovery.subscribe(listener, slotKey);
      return () => {
        lease.current.active = false;
        lease.current.generation += 1;
        unsubscribe();
      };
    },
    [recovery, slotKey]
  );
  const snapshot = useSyncExternalStore(subscribe, recovery.getSnapshot);
  const reload = useWorkbenchReload();
  const { t } = useTranslation();
  const isCurrent = () => {
    const project = owner.getSnapshot().activeProject;
    return (
      isAccountScopeCurrent(account) &&
      project.id === projectId &&
      !!project.widgetInstances[instanceId] &&
      owner.widgetRecovery.get(instanceId) === recovery
    );
  };
  return {
    ...snapshot,
    hasFailure: Object.keys(snapshot.errors).some((name) => name.startsWith(`${region}:`)),
    error: snapshot.errors[key(slot)],
    getError: (name: string) => snapshot.errors[key(name)],
    retry: () => {
      if (isCurrent() && lease.current.active) {
        recovery.retry(widget);
      }
    },
    requestReload: () => {
      if (!isCurrent() || !lease.current.active || !snapshot.reloadAvailable) {
        return;
      }
      const generation = lease.current.generation;
      const retryGeneration = snapshot.generation;
      const current = () =>
        lease.current.active &&
        lease.current.generation === generation &&
        isCurrent() &&
        recovery.getSnapshot().generation === retryGeneration;
      return async () => {
        let refusal: { error: unknown } | undefined;
        if (current()) {
          try {
            await recovery.reload(() => reload(current));
          } catch (error) {
            refusal = { error };
          }
        }
        // An expired request can explain itself only to the same owner after its view is visible again.
        if (!lease.current.active || !isCurrent()) {
          return;
        }
        if (!current()) {
          refusal = { error: new WorkbenchReloadError('inactive') };
        }
        if (!refusal) {
          return;
        }
        const { error } = refusal;
        createLogger({ area: 'widget-recovery', namespace: 'system' })
          .child({ projectId })
          .error({
            error,
            context: error instanceof AggregateError ? { pixelFailures: error.errors } : undefined,
            message: 'Editor reload was refused',
            name: 'widget.reload-refused',
          });
        toaster.create({
          type: 'error',
          title: t('widgets.failure.reloadFailed'),
          description: describeWorkbenchReloadError(error, t),
        });
      };
    },
    recover: () => {
      if (
        isCurrent() &&
        widget.implementation.getStatus() === 'loaded' &&
        recovery.getSnapshot().generation === snapshot.generation
      ) {
        recovery.recover(key(slot));
      }
    },
    fail: (error: Error) => {
      return (
        isCurrent() && recovery.getSnapshot().generation === snapshot.generation && recovery.fail(key(slot), error)
      );
    },
  };
};
