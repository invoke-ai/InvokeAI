import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';

import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { useMountEffect } from '@platform/react/useMountEffect';
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
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getCanvasGallerySaveErrorAction, withMatchingCanvasProject } from './canvasGallerySaveState';

type CreateFromBboxEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'layers' | 'lifecycle' | 'projectId'>;

export const useCreateFromBbox = (
  engine: CreateFromBboxEngine | null
): { isCreating: boolean; createFromBbox: (destination: CreateFromBboxDestination) => Promise<void> } => {
  const { t } = useTranslation();
  const notify = useNotify();
  const queries = useWorkbenchQueries();
  const { canvas, generation, notifications } = useWorkbenchCommands();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const isCreatingRef = useRef(false);
  const [isCreating, setIsCreating] = useState(false);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const createFromBbox = useCallback(
    async (destination: CreateFromBboxDestination): Promise<void> => {
      if (isCreatingRef.current) {
        return;
      }

      const project = queries.getSnapshot().activeProject;

      await withMatchingCanvasProject(engine, project.id, async (matchedEngine) => {
        isCreatingRef.current = true;
        setIsCreating(true);

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
          isCreatingRef.current = false;
          setIsCreating(false);
        }
      });
    },
    [canvas, engine, generation, models, notifications, notify, openWorkbenchWidget, queries, t]
  );

  return { createFromBbox, isCreating };
};
