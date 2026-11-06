import type { RegisteredWidget } from '@workbench/widgetContracts';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
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
  const key = (name: string) => `${region}:${name}`;
  // The registry resource can outlive this view; a reload requested here cannot.
  const lease = useRef({ active: true, generation: 0 });
  const subscribe = useCallback(
    (listener: () => void) => {
      lease.current.active = true;
      const unsubscribe = recovery.subscribe(listener);
      return () => {
        lease.current.active = false;
        lease.current.generation += 1;
        unsubscribe();
      };
    },
    [recovery]
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
        const generation = lease.current.generation;
        const current = () => lease.current.active && lease.current.generation === generation && isCurrent();
        recovery.retry(widget, async () => {
          if (current()) {
            await reload(current).catch((error: unknown) => {
              if (!current()) {
                return;
              }
              toaster.create({
                type: 'error',
                title: t('widgets.failure.reloadFailed'),
                description: error instanceof Error ? error.message : String(error),
              });
            });
          }
        });
      }
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
