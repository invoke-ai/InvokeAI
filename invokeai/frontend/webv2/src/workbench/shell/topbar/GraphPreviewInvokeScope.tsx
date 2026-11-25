import type { GraphPreviewInvokeScopeProps } from '@workbench/widget-frame/graphPreviewInvoke';

import { useCallback, useMemo } from 'react';

import { InvokeButton } from './InvokeButton';
import { type InvocationState, useInvocationState } from './useInvocationState';

/** A graph preview's Invoke: the top bar's control for the previewed source's route, not the active one. */
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
