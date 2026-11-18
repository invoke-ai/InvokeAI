import type { UpscaleUiAdapter } from '@features/upscale';
import type { ReactNode } from 'react';

import { areProjectPromptDraftsEqual, getPromptDraftFromValues } from '@features/generation/settings';
import { createDefaultUpscaleWidgetValues, normalizeUpscaleWidgetValues, UpscaleUiProvider } from '@features/upscale';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
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
  const queries = useWorkbenchQueries();
  const notify = useNotify();
  // Key actions by project, not values, to preserve callback identity while typing.
  const { projectId } = project;
  const patchPromptDraft = useCallback<UpscaleUiAdapter['patchPromptDraft']>(
    (values) => commands.generation.patchPromptDraft(values, 'upscale', projectId),
    [commands, projectId]
  );
  const patchValues = useCallback<UpscaleUiAdapter['patchValues']>(
    (values, origin) =>
      commands.widgets.patchValues(
        'upscale',
        typeof values === 'function'
          ? (current) => values(normalizeUpscaleWidgetValues(current) ?? createDefaultUpscaleWidgetValues())
          : values,
        projectId,
        origin
      ),
    [commands, projectId]
  );
  const readPromptDraft = useCallback<UpscaleUiAdapter['readPromptDraft']>(() => {
    const target = queries.getProject(projectId);

    return getPromptDraftFromValues(target ? getProjectWidgetValues(target, 'generate') : {});
  }, [projectId, queries]);
  const notifications = useMemo<UpscaleUiAdapter['notifications']>(
    () => ({
      info: (title) => notify.info(title),
      undoable: (title, notice) => notify.undoable(title, { ...notice, projectId }),
    }),
    [notify, projectId]
  );
  const reportError = useCallback<UpscaleUiAdapter['reportError']>(
    (message) => commands.notifications.reportError({ area: 'upscale', message, namespace: 'generation' }),
    [commands]
  );
  const adapter = useMemo<UpscaleUiAdapter>(
    () => ({
      ...project,
      notifications,
      patchPromptDraft,
      patchValues,
      readPromptDraft,
      reportError,
      showPromptSyntaxHighlighting,
    }),
    [notifications, patchPromptDraft, patchValues, project, readPromptDraft, reportError, showPromptSyntaxHighlighting]
  );

  return <UpscaleUiProvider adapter={adapter}>{children}</UpscaleUiProvider>;
};
