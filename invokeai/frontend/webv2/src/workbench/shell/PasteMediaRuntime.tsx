import { classifyGalleryUpload } from '@features/gallery/contracts';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Dialog } from '@platform/ui/Dialog';
import { isModalPresent } from '@platform/ui/modalPresence';
import { isEditableHotkeyTarget } from '@workbench/hotkeys/keys';
import { lazy, Suspense, useCallback, useState } from 'react';

export interface PasteMediaRequest {
  ticket: number;
  files: File[];
  returnFocus: HTMLElement | null;
  settle: () => void;
}

// Modal dialogs announce their presence; a paste from inside a non-modal one, such as a popover, also stays with it.
const isInsideDialog = (target: EventTarget | null): boolean =>
  target instanceof Element && target.closest('[role="dialog"], [role="alertdialog"]') !== null;

const PasteMediaDialog = lazy(() =>
  import('./PasteMediaDialog').then((module) => ({ default: module.PasteMediaDialog }))
);
// The dialog is open, and modal, from the paste that requested it, not from when its module arrives.
const PENDING_DIALOG = <Dialog.Pending />;

/**
 * Offer destinations for workbench media paste. Canvas handles its own chord; text fields and open dialogs retain
 * normal paste behavior.
 */
export const PasteMediaRuntime = () => {
  const [request, setRequest] = useState<PasteMediaRequest | null>(null);
  const settle = useCallback(() => setRequest(null), []);
  // Settling closes the dialog; its request stays rendered until the close animation finishes.
  const dialog = useExitRetainedValue(request);

  useMountEffect(() => {
    let ticket = 0;
    const handlePaste = (event: ClipboardEvent) => {
      const clipboard = event.clipboardData;
      if (!clipboard || isModalPresent() || isInsideDialog(event.target)) {
        return;
      }
      if (isEditableHotkeyTarget(event.target) && clipboard.types.includes('text/plain')) {
        return;
      }
      const files = Array.from(clipboard.files).filter((file) => classifyGalleryUpload(file) !== null);
      if (files.length === 0) {
        return;
      }
      event.preventDefault();
      setRequest({
        ticket: ++ticket,
        files,
        returnFocus: document.activeElement instanceof HTMLElement ? document.activeElement : null,
        settle,
      });
    };
    document.addEventListener('paste', handlePaste);
    return () => document.removeEventListener('paste', handlePaste);
  });

  return dialog.value ? (
    <Suspense fallback={dialog.isOpen ? PENDING_DIALOG : null}>
      <PasteMediaDialog
        key={dialog.value.ticket}
        isOpen={dialog.isOpen}
        request={dialog.value}
        onExitComplete={dialog.release}
      />
    </Suspense>
  ) : null;
};
