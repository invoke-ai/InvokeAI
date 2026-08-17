import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/settings';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';
import type { ReactNode } from 'react';

import { createContext, use, useMemo } from 'react';

/**
 * Upscale's UI port. The context is a dependency-direction port (the feature
 * may not import workbench), not a test seam; no second adapter is expected.
 */
export interface UpscaleUiAdapter {
  patchPromptDraft(values: ProjectPromptDraftPatch): void;
  patchValues(values: Partial<UpscaleWidgetValues>, origin?: 'user' | 'system'): void;
  projectId: string;
  promptDraft: ProjectPromptDraft;
  rawValues: Record<string, unknown>;
  reportError(message: string): void;
  showPromptSyntaxHighlighting: boolean;
  touchGalleryImages(): void;
}

/** The adapter's callbacks, which are stable for the lifetime of a project. */
export type UpscaleUiActions = Pick<
  UpscaleUiAdapter,
  'patchPromptDraft' | 'patchValues' | 'reportError' | 'touchGalleryImages'
>;

const UpscaleUiContext = createContext<UpscaleUiAdapter | null>(null);
/**
 * Actions are published separately from the adapter because the adapter's
 * identity changes on every value patch (it carries `rawValues`). Components
 * that only need to *do* something — not read state — subscribe here and so are
 * not re-rendered by a keystroke elsewhere in the form.
 */
const UpscaleUiActionsContext = createContext<UpscaleUiActions | null>(null);

export const UpscaleUiProvider = ({ adapter, children }: { adapter: UpscaleUiAdapter; children: ReactNode }) => {
  const { patchPromptDraft, patchValues, reportError, touchGalleryImages } = adapter;
  const actions = useMemo<UpscaleUiActions>(
    () => ({ patchPromptDraft, patchValues, reportError, touchGalleryImages }),
    [patchPromptDraft, patchValues, reportError, touchGalleryImages]
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
