import type {
  IntermediatesCleanupMode,
  IntermediatesOperation,
  IntermediatesScope,
} from '@features/intermediates/core/types';

import { createIntermediatesPreview, startIntermediatesOperation } from '@features/intermediates/data/api';
import {
  adoptIntermediatesOperation,
  reconcileIntermediatesOperations,
} from '@features/intermediates/data/operationStore';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { ApiError, getApiErrorMessage } from '@platform/transport/http';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ClearDialogState } from './ClearDialog';

/** The confirmation flow: preview a scope, optionally switch to force, and start the operation it described. */
export const useCleanupDialog = ({ onStarted }: { onStarted: () => void }) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [dialog, setDialog] = useState<ClearDialogState | null>(null);
  const triggerRef = useRef<HTMLElement | null>(null);
  const startedRef = useRef(false);
  const requestRef = useRef(0);
  const scopeRef = useRef<IntermediatesScope | null>(null);

  const loadPreview = useCallback(
    async (mode: IntermediatesCleanupMode, scope: IntermediatesScope) => {
      const requestId = ++requestRef.current;
      const owner = captureAccountScope();
      setDialog({ isStarting: false, mode, preview: null, previewError: null, startError: null });
      try {
        const preview = await createIntermediatesPreview({ mode, scope }, owner.signal);
        assertAccountScopeCurrent(owner);
        if (requestRef.current === requestId) {
          setDialog((current) => (current ? { ...current, preview } : current));
        }
      } catch (error) {
        if (requestRef.current === requestId && isAccountScopeCurrent(owner)) {
          setDialog((current) =>
            current
              ? { ...current, previewError: getApiErrorMessage(error, t('intermediates.dialog.previewFailed')) }
              : current
          );
        }
      }
    },
    [t]
  );

  const open = useCallback(
    (scope: IntermediatesScope, trigger: HTMLElement | null, mode: IntermediatesCleanupMode = 'safe') => {
      triggerRef.current = trigger;
      startedRef.current = false;
      scopeRef.current = scope;
      setDialog(null);
      void loadPreview(mode, scope);
    },
    [loadPreview]
  );
  const close = useCallback(() => {
    requestRef.current += 1;
    setDialog(null);
  }, []);
  const retryPreview = useCallback(() => {
    if (dialog && scopeRef.current) {
      void loadPreview(dialog.mode, scopeRef.current);
    }
  }, [dialog, loadPreview]);
  const changeMode = useCallback(
    (mode: IntermediatesCleanupMode) => {
      if (scopeRef.current) {
        void loadPreview(mode, scopeRef.current);
      }
    },
    [loadPreview]
  );
  const confirm = useCallback(async () => {
    const preview = dialog?.preview ?? null;
    if (!dialog || !preview) {
      return;
    }
    const owner = captureAccountScope();
    const requestId = requestRef.current;
    const started = () => {
      startedRef.current = true;
      onStarted();
      setDialog(null);
    };
    setDialog((current) => (current ? { ...current, isStarting: true, startError: null } : current));
    try {
      const operation = await startIntermediatesOperation({ previewId: preview.previewId }, owner.signal);
      assertAccountScopeCurrent(owner);
      adoptIntermediatesOperation(queryClient, owner, operation);
      started();
    } catch (error) {
      if (!isAccountScopeCurrent(owner) || (error instanceof DOMException && error.name === 'AbortError')) {
        setDialog((current) => (current ? { ...current, isStarting: false } : current));
        return;
      }
      if (error instanceof DOMException && error.name === 'TimeoutError') {
        // The server may have accepted the start; if it is running, follow it as if the answer had arrived. A
        // listing that fails too leaves that unknown, which is not the same as "nothing is running".
        let adopted: IntermediatesOperation | null = null;
        let listed = true;
        try {
          adopted = await reconcileIntermediatesOperations(queryClient, owner, { startedAfter: preview.createdAt });
        } catch {
          listed = false;
        }
        if (!isAccountScopeCurrent(owner) || requestRef.current !== requestId) {
          return;
        }
        if (adopted) {
          started();
          return;
        }
        setDialog((current) =>
          current
            ? {
                ...current,
                isStarting: false,
                startError: listed
                  ? t('intermediates.dialog.startTimedOut', {
                      action: t(
                        current.mode === 'force' ? 'intermediates.dialog.forceConfirm' : 'intermediates.dialog.confirm'
                      ),
                    })
                  : t('intermediates.dialog.startUnknown'),
              }
            : current
        );
        return;
      }
      // A 404 means the preview expired or was already confirmed; only a new preview can move forward, so offer
      // that instead of a dead Confirm.
      const previewGone = error instanceof ApiError && error.status === 404;
      setDialog((current) =>
        current
          ? {
              ...current,
              isStarting: false,
              preview: previewGone ? null : current.preview,
              previewError: previewGone ? t('intermediates.dialog.previewExpired') : current.previewError,
              startError: previewGone ? null : getApiErrorMessage(error, t('intermediates.dialog.startFailed')),
            }
          : current
      );
    }
  }, [dialog, onStarted, queryClient, t]);

  const hasStarted = useCallback(() => startedRef.current, []);

  return { changeMode, close, confirm, hasStarted, open, retryPreview, state: dialog, triggerRef };
};
