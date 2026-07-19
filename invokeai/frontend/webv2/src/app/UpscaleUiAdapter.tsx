import type { UpscaleUiAdapter } from '@features/upscale';
import type { ReactNode } from 'react';

import { areProjectPromptDraftsEqual, getPromptDraftFromValues } from '@features/generation/utility';
import { UpscaleUiProvider } from '@features/upscale';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

/**
 * Production binding of Upscale's UI port: maps the upscale widget instance
 * and prompt drafts out of the Workbench aggregate. No second adapter is expected.
 */
export const UpscaleUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const project = useActiveProjectSelector(
    (activeProject) => {
      const instance = Object.values(activeProject.widgetInstances).find((candidate) => candidate.typeId === 'upscale');

      return {
        projectId: activeProject.id,
        promptDraft: getPromptDraftFromValues(getProjectWidgetValues(activeProject, 'generate')),
        rawValues: instance?.state.values ?? {},
        showPromptSyntaxHighlighting: activeProject.settings.showPromptSyntaxHighlighting,
      };
    },
    (left, right) =>
      left.projectId === right.projectId &&
      areProjectPromptDraftsEqual(left.promptDraft, right.promptDraft) &&
      left.rawValues === right.rawValues &&
      left.showPromptSyntaxHighlighting === right.showPromptSyntaxHighlighting
  );
  const commands = useWorkbenchCommands();
  const adapter = useMemo<UpscaleUiAdapter>(
    () => ({
      ...project,
      patchPromptDraft: (values) => commands.generation.patchPromptDraft(values, project.projectId),
      patchValues: (values) => commands.widgets.patchValues('upscale', values, project.projectId),
      reportError: (message) =>
        commands.notifications.reportError({ area: 'upscale', message, namespace: 'generation' }),
      touchGalleryImages: () => commands.gallery.touchImages(),
    }),
    [commands, project]
  );

  return <UpscaleUiProvider adapter={adapter}>{children}</UpscaleUiProvider>;
};
