import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { VideoWidgetValues } from '@features/video/core/types';
import type { ReactNode } from 'react';

import { createContext, use, useMemo } from 'react';

/**
 * Video's UI port. The context is a dependency-direction port (the feature
 * may not import workbench), not a test seam; no second adapter is expected.
 */
export interface VideoUiAdapter {
  patchPromptDraft(values: ProjectPromptDraftPatch): void;
  patchValues(values: Partial<VideoWidgetValues>, origin?: 'user' | 'system'): void;
  projectId: string;
  promptDraft: ProjectPromptDraft;
  rawValues: Record<string, unknown>;
  reportError(message: string): void;
  showPromptSyntaxHighlighting: boolean;
}

/** The adapter's callbacks, which are stable for the lifetime of a project. */
export type VideoUiActions = Pick<VideoUiAdapter, 'patchPromptDraft' | 'patchValues' | 'reportError'>;

const VideoUiContext = createContext<VideoUiAdapter | null>(null);
/**
 * Actions are published separately from the adapter because the adapter's
 * identity changes on every value patch (it carries `rawValues`). Components
 * that only need to *do* something — not read state — subscribe here and so are
 * not re-rendered by a keystroke elsewhere in the form.
 */
const VideoUiActionsContext = createContext<VideoUiActions | null>(null);

export const VideoUiProvider = ({ adapter, children }: { adapter: VideoUiAdapter; children: ReactNode }) => {
  const { patchPromptDraft, patchValues, reportError } = adapter;
  const actions = useMemo<VideoUiActions>(
    () => ({ patchPromptDraft, patchValues, reportError }),
    [patchPromptDraft, patchValues, reportError]
  );

  return (
    <VideoUiActionsContext value={actions}>
      <VideoUiContext value={adapter}>{children}</VideoUiContext>
    </VideoUiActionsContext>
  );
};

export const useVideoUi = (): VideoUiAdapter => {
  const adapter = use(VideoUiContext);

  if (!adapter) {
    throw new Error('Video UI requires an App-composed VideoUiProvider.');
  }

  return adapter;
};

export const useVideoUiActions = (): VideoUiActions => {
  const actions = use(VideoUiActionsContext);

  if (!actions) {
    throw new Error('Video UI requires an App-composed VideoUiProvider.');
  }

  return actions;
};
