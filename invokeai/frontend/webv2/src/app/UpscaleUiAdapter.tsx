import type { UpscaleUiAdapter } from '@features/upscale';
import type { ReactNode } from 'react';

import { areProjectPromptDraftsEqual, getPromptDraftFromValues } from '@features/generation/settings';
import { UpscaleUiProvider } from '@features/upscale';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';

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
  // Syntax highlighting is an account preference, not project data.
  const showPromptSyntaxHighlighting = useWorkbenchPreferenceSelector(
    (preferences) => preferences.showPromptSyntaxHighlighting
  );
  const commands = useWorkbenchCommands();
  // Key actions by project, not values, to preserve callback identity while typing.
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
  const adapter = useMemo<UpscaleUiAdapter>(
    () => ({
      ...project,
      patchPromptDraft,
      patchValues,
      reportError,
      showPromptSyntaxHighlighting,
    }),
    [patchPromptDraft, patchValues, project, reportError, showPromptSyntaxHighlighting]
  );

  return <UpscaleUiProvider adapter={adapter}>{children}</UpscaleUiProvider>;
};
