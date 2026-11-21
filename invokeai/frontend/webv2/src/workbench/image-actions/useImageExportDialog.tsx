import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ReactNode } from 'react';

import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { Dialog } from '@platform/ui/Dialog';
import { lazy, Suspense, useCallback, useRef, useState } from 'react';

import type { ExecuteImageExport, ImageExportTarget, RequestImageExport } from './imageExport';

export interface ImageExportDialogRequest {
  execute: ExecuteImageExport;
  owner: AccountScope;
  returnFocus: () => HTMLElement | null;
  target: ImageExportTarget;
  ticket: number;
}

/** The dialog's controls and preview load on first open, outside the editor's initial payload. */
const LazyImageExportDialog = lazy(() =>
  import('./ImageExportDialog').then((module) => ({ default: module.ImageExportDialog }))
);

// The dialog is open, and modal, from the request that opened it, not from when its module arrives.
const PENDING_DIALOG = <Dialog.Pending />;

/** One export dialog per host; a new request replaces an open one, and an account change closes it. */
export const useImageExportDialog = (): { dialog: ReactNode; requestImageExport: RequestImageExport } => {
  const [request, setRequest] = useState<ImageExportDialogRequest | null>(null);
  const retained = useExitRetainedValue(request);
  const ticketRef = useRef(0);
  const releaseRef = useRef<(() => void) | null>(null);

  const close = useCallback(() => {
    releaseRef.current?.();
    releaseRef.current = null;
    setRequest(null);
  }, []);

  useMountEffect(() => () => releaseRef.current?.());

  const requestImageExport = useCallback<RequestImageExport>(
    (target, execute, options = {}) => {
      const owner = captureAccountScope();
      const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      const ticket = (ticketRef.current += 1);

      releaseRef.current?.();
      owner.signal.addEventListener('abort', close, { once: true });
      releaseRef.current = () => owner.signal.removeEventListener('abort', close);
      setRequest({
        execute,
        owner,
        // A menu item that opened the dialog is gone by the time it closes; the dialog then restores focus itself.
        returnFocus: () => options.returnFocus?.() ?? (opener?.isConnected ? opener : null),
        target,
        ticket,
      });
    },
    [close]
  );

  return {
    dialog: retained.value ? (
      <Suspense fallback={retained.isOpen ? PENDING_DIALOG : null}>
        <LazyImageExportDialog
          key={retained.value.ticket}
          isOpen={retained.isOpen}
          request={retained.value}
          onClose={close}
          onExitComplete={retained.release}
        />
      </Suspense>
    ) : null,
    requestImageExport,
  };
};
