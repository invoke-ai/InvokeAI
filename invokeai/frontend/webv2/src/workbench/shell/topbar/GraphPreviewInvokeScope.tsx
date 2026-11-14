import type { GraphPreviewInvokeScopeProps } from '@workbench/widget-frame/graphPreviewInvoke';

import { useCallback, useMemo } from 'react';

import { InvokeButton } from './InvokeButton';
import { type InvocationState, useInvocationState } from './useInvocationState';

/**
 * A graph preview's Invoke: this top bar's control, for the previewed source's route rather than the active one, its
 * issue card, fixes, preparing state and iterations included. The route is the preview's, so the card shows it without
 * a picker or lock, and the control takes a dialog footer's button size. A run that starts, or an issue's action that
 * leads elsewhere, closes the preview. The shell provides it to widget frames (`GraphPreviewInvokeScopeContext`).
 */
export const GraphPreviewInvokeScope = ({ render, sourceId, onOpenChange }: GraphPreviewInvokeScopeProps) => {
  const state = useInvocationState({ sourceId });
  const close = useCallback(() => onOpenChange(false), [onOpenChange]);
  const { invoke, isValid } = state;
  const previewState = useMemo<InvocationState>(
    () => ({ ...state, invoke: (options) => invoke({ ...options, onStarted: close }) }),
    [close, invoke, state]
  );
  const slot = useMemo(
    () => ({
      control: <InvokeButton isRouteFixed size="md" state={previewState} onJump={close} />,
      isReady: isValid,
    }),
    [close, isValid, previewState]
  );

  return render(slot);
};
