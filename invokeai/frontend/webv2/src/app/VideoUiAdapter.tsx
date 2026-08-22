import type { VideoUiAdapter } from '@features/video';
import type { ReactNode } from 'react';

import { areProjectPromptDraftsEqual, getPromptDraftFromValues } from '@features/generation/settings';
import { VideoUiProvider } from '@features/video';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';

/**
 * Production binding of Video's UI port: maps the video widget instance and
 * the shared prompt draft out of the Workbench aggregate. No second adapter is
 * expected.
 */
export const VideoUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const project = useActiveProjectSelector(
    (activeProject) => {
      const instance = Object.values(activeProject.widgetInstances).find((candidate) => candidate.typeId === 'video');

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
  // The port's callbacks are keyed to the project, not to its contents: rebuilding
  // them whenever `rawValues` changes would hand every consumer new function
  // identities on each keystroke, re-rendering memoized fields that did not change.
  const { projectId } = project;
  const patchPromptDraft = useCallback<VideoUiAdapter['patchPromptDraft']>(
    (values) => commands.generation.patchPromptDraft(values, 'video', projectId),
    [commands, projectId]
  );
  const patchValues = useCallback<VideoUiAdapter['patchValues']>(
    (values, origin) => commands.widgets.patchValues('video', values, projectId, origin),
    [commands, projectId]
  );
  const reportError = useCallback<VideoUiAdapter['reportError']>(
    (message) => commands.notifications.reportError({ area: 'video', message, namespace: 'generation' }),
    [commands]
  );
  const adapter = useMemo<VideoUiAdapter>(
    () => ({
      ...project,
      patchPromptDraft,
      patchValues,
      reportError,
      showPromptSyntaxHighlighting,
    }),
    [patchPromptDraft, patchValues, project, reportError, showPromptSyntaxHighlighting]
  );

  return <VideoUiProvider adapter={adapter}>{children}</VideoUiProvider>;
};
