import type { UpscaleUiAdapter } from '@features/upscale';
import type { ReactNode } from 'react';

import { invalidateGallery } from '@features/gallery/queries';
import { areProjectPromptDraftsEqual, getPromptDraftFromValues } from '@features/generation/settings';
import { UpscaleUiProvider } from '@features/upscale';
import { useQueryClient } from '@tanstack/react-query';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';

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
      };
    },
    (left, right) =>
      left.projectId === right.projectId &&
      areProjectPromptDraftsEqual(left.promptDraft, right.promptDraft) &&
      left.rawValues === right.rawValues
  );
  // Syntax highlighting is a per-user preference, not a property of the
  // project, so it is joined here rather than read off the document.
  const showPromptSyntaxHighlighting = useWorkbenchPreferenceSelector(
    (preferences) => preferences.showPromptSyntaxHighlighting
  );
  const commands = useWorkbenchCommands();
  const queryClient = useQueryClient();
  // The port's callbacks are keyed to the project, not to its contents: rebuilding
  // them whenever `rawValues` changes would hand every consumer new function
  // identities on each keystroke, re-rendering memoized fields that did not change.
  const { projectId } = project;
  const patchPromptDraft = useCallback<UpscaleUiAdapter['patchPromptDraft']>(
    (values) => commands.generation.patchPromptDraft(values, 'upscale', projectId),
    [commands, projectId]
  );
  const patchValues = useCallback<UpscaleUiAdapter['patchValues']>(
    (values, origin) => commands.widgets.patchValues('upscale', values, projectId, origin),
    [commands, projectId]
  );
  const reportError = useCallback<UpscaleUiAdapter['reportError']>(
    (message) => commands.notifications.reportError({ area: 'upscale', message, namespace: 'generation' }),
    [commands]
  );
  const touchGalleryImages = useCallback(() => void invalidateGallery(queryClient), [queryClient]);
  const adapter = useMemo<UpscaleUiAdapter>(
    () => ({
      ...project,
      patchPromptDraft,
      patchValues,
      reportError,
      showPromptSyntaxHighlighting,
      touchGalleryImages,
    }),
    [patchPromptDraft, patchValues, project, reportError, showPromptSyntaxHighlighting, touchGalleryImages]
  );

  return <UpscaleUiProvider adapter={adapter}>{children}</UpscaleUiProvider>;
};
