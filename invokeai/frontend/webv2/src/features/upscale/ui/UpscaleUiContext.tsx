import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';
import type { ReactNode } from 'react';

import { createContext, use, useMemo } from 'react';

/** This UI port preserves dependency direction: Upscale cannot import Workbench. */
export interface UpscaleUiAdapter {
  /** Notices about this project; `undoable` reports a change that already happened, with its Undo. */
  notifications: {
    info(title: string): void;
    undoable(title: string, notice: { onUndo: () => void; supersedes: string }): void;
  };
  patchPromptDraft(values: ProjectPromptDraftPatch): void;
  /** Functional patches run synchronously against the originating project's latest normalized values. */
  patchValues(
    values: Partial<UpscaleWidgetValues> | ((current: UpscaleWidgetValues) => Partial<UpscaleWidgetValues>),
    origin?: 'user' | 'system'
  ): void;
  projectId: string;
  promptDraft: ProjectPromptDraft;
  rawValues: Record<string, unknown>;
  /** This project's shared prompt draft at call time, for an Undo that checks what changed since. */
  readPromptDraft(): ProjectPromptDraft;
  reportError(message: string): void;
  showPromptSyntaxHighlighting: boolean;
}

/** The adapter's callbacks, which are stable for the lifetime of a project. */
export type UpscaleUiActions = Pick<
  UpscaleUiAdapter,
  'notifications' | 'patchPromptDraft' | 'patchValues' | 'readPromptDraft' | 'reportError'
>;

const UpscaleUiContext = createContext<UpscaleUiAdapter | null>(null);
/** Separate stable actions from value-bearing adapters so action-only consumers do not rerender on form keystrokes. */
const UpscaleUiActionsContext = createContext<UpscaleUiActions | null>(null);

export const UpscaleUiProvider = ({ adapter, children }: { adapter: UpscaleUiAdapter; children: ReactNode }) => {
  const { notifications, patchPromptDraft, patchValues, readPromptDraft, reportError } = adapter;
  const actions = useMemo<UpscaleUiActions>(
    () => ({ notifications, patchPromptDraft, patchValues, readPromptDraft, reportError }),
    [notifications, patchPromptDraft, patchValues, readPromptDraft, reportError]
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
