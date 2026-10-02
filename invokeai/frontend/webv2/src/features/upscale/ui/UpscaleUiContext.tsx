import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';
import type { ReactNode } from 'react';

import { createContext, use, useMemo } from 'react';

/** This UI port preserves dependency direction: Upscale cannot import Workbench. */
export interface UpscaleUiAdapter {
  patchPromptDraft(values: ProjectPromptDraftPatch): void;
  patchValues(values: Partial<UpscaleWidgetValues>, origin?: 'user' | 'system'): void;
  projectId: string;
  promptDraft: ProjectPromptDraft;
  rawValues: Record<string, unknown>;
  reportError(message: string): void;
  showPromptSyntaxHighlighting: boolean;
}

/** The adapter's callbacks, which are stable for the lifetime of a project. */
export type UpscaleUiActions = Pick<UpscaleUiAdapter, 'patchPromptDraft' | 'patchValues' | 'reportError'>;

const UpscaleUiContext = createContext<UpscaleUiAdapter | null>(null);
/** Separate stable actions from value-bearing adapters so action-only consumers do not rerender on form keystrokes. */
const UpscaleUiActionsContext = createContext<UpscaleUiActions | null>(null);

export const UpscaleUiProvider = ({ adapter, children }: { adapter: UpscaleUiAdapter; children: ReactNode }) => {
  const { patchPromptDraft, patchValues, reportError } = adapter;
  const actions = useMemo<UpscaleUiActions>(
    () => ({ patchPromptDraft, patchValues, reportError }),
    [patchPromptDraft, patchValues, reportError]
  );

  return (
    <UpscaleUiActionsContext value={actions}>
      <UpscaleUiContext value={adapter}>{children}</UpscaleUiContext>
    </UpscaleUiActionsContext>
  );
};

export const useUpscaleUi = (): UpscaleUiAdapter => {
  const adapter = use(UpscaleUiContext);

  if (!adapter) {
    throw new Error('Upscale UI requires an App-composed UpscaleUiProvider.');
  }

  return adapter;
};

export const useUpscaleUiActions = (): UpscaleUiActions => {
  const actions = use(UpscaleUiActionsContext);

  if (!actions) {
    throw new Error('Upscale UI requires an App-composed UpscaleUiProvider.');
  }

  return actions;
};
