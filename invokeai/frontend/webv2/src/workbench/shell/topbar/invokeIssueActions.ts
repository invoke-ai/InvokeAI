import type { GenerationIssue } from '@features/generation/contracts';

import { flushGenerateDrafts, focusPositivePrompt, revealGenerateIssue } from '@features/generation/react';
import {
  describeGenerationIssueFix,
  getGenerationIssueFixPatch,
  normalizeGenerateSettings,
} from '@features/generation/settings';
import { useCapabilities } from '@features/identity';
import { openModelManager } from '@features/models';
import { revealWorkflowNode } from '@features/workflow/react';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { getWidgetTypeIdForSourceId } from '@workbench/graphWidgets';
import { patchWorkbenchPreferences, getWorkbenchPreferences } from '@workbench/settings/store';
import { openWorkbenchWidget, revealOrOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { useRevealLayerProperties } from '@workbench/widgets/layers/revealLayerProperties';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationIssue } from './invokeIssueTargets';

export interface InvokeIssueAction {
  id: string;
  label: string;
  run: () => void;
  /** Acts in place (a fix, a destination switch), so the issue list stays open to show what is left. */
  stays?: boolean;
}

/** Issues the Model Manager fixes: no usable model at all, or the chosen one is gone. */
const isMissingModelIssue = (issue: GenerationIssue): boolean =>
  issue.id === 'model:unsupported' || issue.id.startsWith('model:notInstalled:');

/**
 * The ways out of each Invoke issue: a one-step fix where Generate has one, then a jump to where the problem is fixed
 * (the Generate section, the canvas layer, the workflow node, the Model Manager, the widget). A lost connection has
 * none: it resolves itself on reconnect.
 */
export const useInvokeIssueActions = (): ((issue: InvocationIssue) => InvokeIssueAction[]) => {
  const { t } = useTranslation();
  const { generation, widgets } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const focus = useWorkbenchFocus();
  const { getWidgetById, getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const revealLayerProperties = useRevealLayerProperties();
  const { canManageModels } = useCapabilities();

  return useCallback(
    ({ target }: InvocationIssue): InvokeIssueAction[] => {
      const reveal = (widgetId: string) =>
        revealOrOpenWorkbenchWidget({ focus, getWidgetsForRegion, queries, widgets }, widgetId);
      const show = (run: () => void): InvokeIssueAction => ({
        id: 'show',
        label: t('topbar.invoke.actions.show'),
        run,
      });

      switch (target.kind) {
        case 'connection':
          return [];
        case 'generation': {
          const { issue } = target;
          const projectId = queries.getSnapshot().activeProject.id;
          const actions: InvokeIssueAction[] = [];

          if (issue.fix) {
            const { fix } = issue;

            actions.push({
              id: 'fix',
              label: describeGenerationIssueFix(t, fix),
              stays: true,
              run: () => {
                flushGenerateDrafts();
                const project = queries.getProject(projectId);
                const settings = project
                  ? normalizeGenerateSettings(getProjectWidgetValues(project, 'generate'))
                  : null;

                if (settings) {
                  generation.patchSettings(getGenerationIssueFixPatch(settings, fix), projectId);
                }
              },
            });
          }
          if (isMissingModelIssue(issue) && canManageModels) {
            actions.push({
              id: 'models',
              label: t('topbar.invoke.actions.openModelManager'),
              run: () => openModelManager({ modelType: 'main', projectId }),
            });
          }
          actions.push(
            show(() => {
              if (issue.target.section !== 'model') {
                void patchWorkbenchPreferences({
                  generateSectionsOpen: {
                    ...getWorkbenchPreferences().generateSectionsOpen,
                    [issue.target.section]: true,
                  },
                });
              }
              reveal('generate');
              revealGenerateIssue(issue);
            })
          );

          return actions;
        }
        case 'prompt':
          return [
            show(() => {
              reveal('generate');
              requestAnimationFrame(() => focusPositivePrompt());
            }),
          ];
        case 'layer':
          return [
            {
              id: 'show',
              label: t('topbar.invoke.actions.showLayer'),
              run: () => revealLayerProperties(target.layerId),
            },
          ];
        case 'workflowNode':
          return [
            {
              id: 'show',
              label: t('topbar.invoke.actions.showNode'),
              run: () => {
                // The editor shows in the center; a side panel holds the workflow's form, which has no graph to select.
                const opened = openWorkbenchWidget(widgets, getWidgetsForRegion, 'workflow', {
                  preferredRegions: ['center'],
                  requireCenterView: true,
                });

                // Without an editor to show it in, a selection request would wait for one that may never mount.
                if (opened.ok) {
                  focus.focusRegion(opened.region, 'workflow');
                  revealWorkflowNode(target.nodeId);
                }
              },
            },
          ];
        case 'mountSource': {
          const widgetId = getWidgetTypeIdForSourceId(target.sourceId);
          const widget = getWidgetById(widgetId);

          return widget
            ? [
                {
                  id: 'add',
                  label: t('topbar.invoke.actions.addWidget', { label: resolveWidgetLabel(widget.manifest, t) }),
                  run: () => reveal(widgetId),
                },
              ]
            : [];
        }
        case 'destination':
          return [
            {
              id: 'destination',
              label: t('topbar.invoke.actions.sendToGallery'),
              run: () => generation.setDestination(target.destination),
              stays: true,
            },
          ];
        case 'widget':
          return [show(() => reveal(target.widgetId))];
      }
    },
    [canManageModels, focus, generation, getWidgetById, getWidgetsForRegion, queries, revealLayerProperties, t, widgets]
  );
};
