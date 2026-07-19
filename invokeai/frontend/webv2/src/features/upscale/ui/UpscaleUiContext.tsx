import type { ProjectPromptDraft, ProjectPromptDraftPatch } from '@features/generation/utility';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';
import type { ReactNode } from 'react';

import { createContext, use } from 'react';

export interface UpscaleUiAdapter {
  patchPromptDraft(values: ProjectPromptDraftPatch): void;
  patchValues(values: Partial<UpscaleWidgetValues>): void;
  projectId: string;
  promptDraft: ProjectPromptDraft;
  rawValues: Record<string, unknown>;
  reportError(message: string): void;
  showPromptSyntaxHighlighting: boolean;
  touchGalleryImages(): void;
}

const UpscaleUiContext = createContext<UpscaleUiAdapter | null>(null);

export const UpscaleUiProvider = ({ adapter, children }: { adapter: UpscaleUiAdapter; children: ReactNode }) => (
  <UpscaleUiContext value={adapter}>{children}</UpscaleUiContext>
);

export const useUpscaleUi = (): UpscaleUiAdapter => {
  const adapter = use(UpscaleUiContext);

  if (!adapter) {
    throw new Error('Upscale UI requires an App-composed UpscaleUiProvider.');
  }

  return adapter;
};
