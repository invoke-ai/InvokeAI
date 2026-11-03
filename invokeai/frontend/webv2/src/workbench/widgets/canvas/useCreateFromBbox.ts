import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { useMountEffect } from '@platform/react/useMountEffect';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createKeyedTransientStore } from '@platform/state/externalStore';
import {
  createFromBbox as runCreateFromBbox,
  getCreateFromBboxNotice,
  type CreateFromBboxDestination,
} from '@workbench/canvas-operations/api';
import { appendReferenceImage } from '@workbench/image-actions/appendReferenceImage';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { getCanvasGallerySaveErrorAction, withMatchingCanvasProject } from './canvasGallerySaveState';
import { describeAdjustmentRefusal } from './useStructuralCommit';

type CreateFromBboxEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'layers' | 'lifecycle' | 'projectId'>;

/** Projects with a create in flight: the context menu and the header's Use frame menu share the gate and busy state. */
const creatingProjects = createKeyedTransientStore<string, true>();
registerAccountOwnedResource({ clear: () => creatingProjects.clear(), name: 'canvas-create-from-bbox' });

export const useCreateFromBbox = (
  engine: CreateFromBboxEngine | null
): { isCreating: boolean; createFromBbox: (destination: CreateFromBboxDestination) => Promise<void> } => {
  const { t } = useTranslation();
  const notify = useNotify();
  const queries = useWorkbenchQueries();
  const { canvas, generation, notifications } = useWorkbenchCommands();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const isCreating = creatingProjects.useValue(engine?.projectId ?? '') === true;

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const createFromBbox = useCallback(
    async (destination: CreateFromBboxDestination): Promise<void> => {
      const project = queries.getSnapshot().activeProject;
      if (creatingProjects.get(project.id)) {
        return;
      }

      await withMatchingCanvasProject(engine, project.id, async (matchedEngine) => {
        creatingProjects.set(project.id, true);

        try {
          const result = await runCreateFromBbox({
            applyCanvasMutation: canvas.apply,
            destination,
            engine: matchedEngine,
            getProject: queries.getProject,
            isActiveProject: queries.isActiveProject,
            project,
          });

          if (result.status === 'uploaded') {
            const latestProject = queries.getProject(project.id);
            if (!latestProject) {
              return;
            }

            const appended = appendReferenceImage({
              generateValues: getProjectWidgetValues(latestProject, 'generate'),
              image: result.image,
              models,
            });

            if (appended.status === 'appended') {
              generation.patchSettings({ referenceImages: appended.referenceImages }, project.id);
              notify.success(t('widgets.canvas.contextMenu.referenceImageAdded'));
              openWorkbenchWidget('generate', { preferredRegions: ['left'] });
            } else if (appended.status === 'full') {
              notify.info(t('widgets.canvas.contextMenu.referenceImageFull'));
            } else {
              notify.info(t('widgets.canvas.contextMenu.referenceImageUnsupported'));
            }
            return;
          }

          const notice = getCreateFromBboxNotice(result);
          if ('refusal' in notice) {
            notify.error(t('widgets.canvas.contextMenu.createError'), describeAdjustmentRefusal(notice.refusal, t));
            return;
          }
          const options =
            notice.options && 'destination' in notice.options
              ? { destination: t(notice.options.destination) }
              : (notice.options ?? {});
          notify[notice.kind](t(notice.titleKey, options));
        } catch (error: unknown) {
          notifications.reportError(
            getCanvasGallerySaveErrorAction(
              error,
              project.id,
              t('widgets.canvas.contextMenu.createError'),
              'canvas-create-from-bbox'
            )
          );
        } finally {
          creatingProjects.delete(project.id);
        }
      });
    },
    [canvas, engine, generation, models, notifications, notify, openWorkbenchWidget, queries, t]
  );

  return { createFromBbox, isCreating };
};
