import type { InvocationRoute } from '@workbench/invocationContracts';
import type { LayoutUndoNotice, Project, UndoRedoEntry } from '@workbench/projectContracts';
import type { TFunction } from 'i18next';

import { useMountEffect } from '@platform/react/useMountEffect';
import { ROUTE_LABEL_KEYS } from '@workbench/invocation';
import { getLayoutPresetLabel } from '@workbench/layoutPresets';
import { resolveSavedLayoutPreset } from '@workbench/layoutPresetSnapshots';
import { withdrawNotificationUndos } from '@workbench/shell/notifications/undoRegistry';
import { useNotify } from '@workbench/useNotify';
import { resolveWidgetInstanceLabel } from '@workbench/widgetLabels';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSubscription } from '@workbench/WorkbenchContext';
import i18next from 'i18next';

const isSameRoute = (left: InvocationRoute, right: InvocationRoute): boolean =>
  left.sourceId === right.sourceId && left.destination === right.destination;

/** The toast's text for a step, worded from the project as the step left it. */
export const describeLayoutUndoNotice = (
  notice: LayoutUndoNotice,
  entry: Pick<UndoRedoEntry, 'project'>,
  project: Project,
  account: Parameters<typeof resolveSavedLayoutPreset>[0],
  t: TFunction
): { message?: string; title: string } => {
  if (notice.kind === 'removeWidget') {
    const instance = entry.project.widgetInstances[notice.instanceId] ?? project.widgetInstances[notice.instanceId];
    const widget = instance ? getWidgetById(instance.typeId) : undefined;
    const label = instance && widget ? resolveWidgetInstanceLabel(instance, widget.manifest, t) : notice.instanceId;

    return { message: t('shell.layoutUndo.removedWidgetHint'), title: t('shell.layoutUndo.removedWidget', { label }) };
  }

  const preset = getLayoutPresetLabel(resolveSavedLayoutPreset(account, notice.presetId), t);
  const routeMessage = isSameRoute(entry.project.invocation, project.invocation)
    ? undefined
    : t('shell.layoutUndo.routeChanged', {
        destination: t(ROUTE_LABEL_KEYS[project.invocation.destination]),
        source: t(ROUTE_LABEL_KEYS[project.invocation.sourceId]),
      });
  const title =
    notice.kind === 'switchPreset'
      ? t('shell.layoutUndo.switchedPreset', { preset })
      : notice.kind === 'revert'
        ? t('shell.layoutUndo.reverted', { preset })
        : t('shell.layoutUndo.discardedCopy', { preset });

  return routeMessage === undefined ? { title } : { message: routeMessage, title };
};

/**
 * Reports each layout step that carries a notice (revert, discarding a preset's changes, switching presets, removing a
 * widget) with an Undo toast, whichever control took it. The toast's Undo acts on its step only while that step is the
 * project's latest: once the step is undone another way (the layout undo command) or another step lands on top, the
 * toast is withdrawn, since undoing it then would take the later work with it. A preset switch is navigation, so its
 * toast stays out of the notification center; the others are recorded there too.
 */
export const LayoutUndoNotices = () => {
  const subscribe = useWorkbenchSubscription();
  const queries = useWorkbenchQueries();
  const { layout } = useWorkbenchCommands();
  const notify = useNotify();

  useMountEffect(() => {
    // Each project's history as last seen: a step is new when it was in none of them. Steps already there when the
    // shell mounts were reported, or never will be.
    const histories = new Map(queries.getSnapshot().projects.map((project) => [project.id, project.undoRedo]));
    // The step each project's Undo toast stands for, while one shows.
    const shownSteps = new Map<string, string>();

    return subscribe(() => {
      const { account, activeProject, projects } = queries.getSnapshot();

      for (const projectId of histories.keys()) {
        if (!projects.some((project) => project.id === projectId)) {
          histories.delete(projectId);
          shownSteps.delete(projectId);
        }
      }

      for (const project of projects) {
        const previous = histories.get(project.id);

        if (previous === project.undoRedo) {
          continue;
        }

        histories.set(project.id, project.undoRedo);
        const top = project.undoRedo.past.at(-1);
        const shownStep = shownSteps.get(project.id);

        if (shownStep !== undefined && top?.id !== shownStep) {
          withdrawNotificationUndos(`layout:${project.id}`);
          shownSteps.delete(project.id);
        }

        // An undo uncovers an older step and a merge keeps the top one: both were seen. A redo's new step has no
        // notice. A step in a project that is not on screen is marked seen without a toast.
        const isNewStep = top !== undefined && previous !== undefined && !previous.past.some(({ id }) => id === top.id);

        if (!isNewStep || !top.notice || project.id !== activeProject.id) {
          continue;
        }

        const projectId = project.id;
        const entryId = top.id;
        // Read at call time, so the toast follows a language changed since the shell mounted.
        const { message, title } = describeLayoutUndoNotice(top.notice, top, project, account, i18next.t);

        notify.undoable(title, {
          ...(message === undefined ? {} : { message }),
          onUndo: () => layout.undo(projectId, entryId),
          projectId,
          supersedes: `layout:${projectId}`,
          transient: top.notice.kind === 'switchPreset',
        });
        shownSteps.set(projectId, entryId);
      }
    });
  });

  return null;
};
