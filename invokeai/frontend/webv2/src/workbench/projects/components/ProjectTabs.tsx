/* eslint-disable react/react-compiler */
import type { Project, QueueItem, WidgetRegion } from '@workbench/types';

import { Icon, Menu, Portal, Separator } from '@chakra-ui/react';
import { useModelLoads } from '@workbench/backend/modelLoadStore';
import { useQueueItemProgress } from '@workbench/backend/progressStore';
import { QueueCircularProgress } from '@workbench/components/QueueProgressIndicator';
import {
  CloseButton,
  IconButton,
  ConfirmDialog,
  MenuContent,
  RenameDialog,
  Tabs,
  Tooltip,
} from '@workbench/components/ui';
import { exportOpenProject } from '@workbench/projects/projectFile';
import { useProjectActions } from '@workbench/projects/useProjectActions';
import { getProjectQueueIndicatorState } from '@workbench/queueSummary';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { flushGenerateDrafts } from '@workbench/widgets/generate/generateDraftRegistry';
import {
  useActiveProjectSelector,
  useWorkbenchDispatch,
  useWorkbenchSelector,
  useWorkbenchStore,
} from '@workbench/WorkbenchContext';
import {
  FileDownIcon,
  FolderCogIcon,
  FolderIcon,
  FolderOpenIcon,
  PencilIcon,
  PlusIcon,
  Trash2Icon,
  XIcon,
} from 'lucide-react';
import { useCallback, useMemo, useRef, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { OpenProjectDialog } from './OpenProjectDialog';

interface ProjectTabSummary {
  id: string;
  name: string;
  queueItems: QueueItem[];
}

const areProjectTabSummariesEqual = (
  left: readonly ProjectTabSummary[],
  right: readonly ProjectTabSummary[]
): boolean =>
  left.length === right.length &&
  left.every(
    (summary, index) =>
      summary.id === right[index]?.id &&
      summary.name === right[index]?.name &&
      summary.queueItems === right[index]?.queueItems
  );

const selectProjectTabSummaries = (projects: readonly Project[]): ProjectTabSummary[] =>
  projects.map(({ id, name, queue }) => ({ id, name, queueItems: queue.items }));

const deleteMenuItemHover = { bg: 'bg.error', color: 'fg.error' } as const;

/**
 * Document-style tabs for the open projects (the session), immediately right
 * of the Invoke control. Tabs are not the library: closing one only removes
 * it from the session — the project stays saved and reopens through the
 * folder button's Open Project dialog. Closing the last tab lands on Home.
 *
 * Each tab is a container with two real buttons (select + close) rather than
 * the prototype's invalid button-nested-in-button, so keyboard focus and
 * click targets behave correctly. Right-clicking a tab opens a context menu
 * with rename, details, export, close, and delete.
 */
export const ProjectTabs = () => {
  const { t } = useTranslation();
  const projectTabSummaries = useWorkbenchSelector(
    (snapshot) => selectProjectTabSummaries(snapshot.state.projects),
    areProjectTabSummariesEqual
  );
  const activeProjectId = useActiveProjectSelector((project) => project.id);
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection.status);
  const modelLoads = useModelLoads();
  const store = useWorkbenchStore();
  const dispatch = useWorkbenchDispatch();
  const { closeProject, deleteProject } = useProjectActions();
  const [menuTarget, setMenuTarget] = useState<{ project: ProjectTabSummary; x: number; y: number } | null>(null);
  const [renameTarget, setRenameTarget] = useState<ProjectTabSummary | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ProjectTabSummary | null>(null);
  const [isOpenDialogVisible, setIsOpenDialogVisible] = useState(false);

  const getProject = useCallback(
    (projectId: string): Project | null =>
      store.getState().projects.find((project) => project.id === projectId) ?? null,
    [store]
  );

  const onSwitchProject = useCallback(
    (projectId: string) => {
      flushGenerateDrafts();
      dispatch({ projectId, type: 'switchProject' });
    },
    [dispatch]
  );

  const openContextMenu = useCallback((project: ProjectTabSummary, event: MouseEvent) => {
    event.preventDefault();
    setMenuTarget({ project, x: event.clientX, y: event.clientY });
  }, []);

  const createProject = useCallback(() => {
    flushGenerateDrafts();
    dispatch({ type: 'createProject' });
  }, [dispatch]);

  const showOpenDialog = useCallback(() => setIsOpenDialogVisible(true), []);
  const hideOpenDialog = useCallback(() => setIsOpenDialogVisible(false), []);

  const closeMenu = useCallback(() => setMenuTarget(null), []);
  const closeRenameDialog = useCallback(() => setRenameTarget(null), []);
  const closeDeleteDialog = useCallback(() => setDeleteTarget(null), []);

  const startRename = useCallback((project: ProjectTabSummary) => setRenameTarget(project), []);
  const startDelete = useCallback((project: ProjectTabSummary) => setDeleteTarget(project), []);

  const closeProjectBySummary = useCallback(
    (project: ProjectTabSummary) => {
      const currentProject = getProject(project.id);

      if (currentProject) {
        closeProject(currentProject);
      }
    },
    [closeProject, getProject]
  );

  const renameProject = useCallback(
    (name: string) => {
      if (renameTarget) {
        dispatch({ name, projectId: renameTarget.id, type: 'renameProject' });
      }
    },
    [dispatch, renameTarget]
  );

  const confirmDeleteProject = useCallback(async () => {
    if (deleteTarget) {
      const currentProject = getProject(deleteTarget.id);

      if (currentProject) {
        await deleteProject(currentProject);
      }
    }
  }, [deleteProject, deleteTarget, getProject]);

  return (
    <>
      <Tabs.Root minW="max-content" variant="subtle" value={activeProjectId} h="full" w="full">
        <Tabs.List flex="1 1 auto" h="full" py="1">
          {projectTabSummaries.map((project) => (
            <ProjectTab
              key={project.id}
              backendConnectionStatus={backendConnectionStatus}
              isActive={project.id === activeProjectId}
              loadingModelsCount={modelLoads.length}
              project={project}
              onCloseProject={closeProjectBySummary}
              onContextMenu={openContextMenu}
              onSwitchProject={onSwitchProject}
            />
          ))}

          <Separator orientation="vertical" h={5} mx="1" alignSelf="center" />

          <Tooltip content={t('projects.createNewProject')} showArrow>
            <IconButton
              aria-label={t('projects.createNewProject')}
              flexShrink={0}
              size="xs"
              variant="ghost"
              alignSelf="center"
              onClick={createProject}
            >
              <PlusIcon />
            </IconButton>
          </Tooltip>

          <Tooltip content={t('projects.openProject')} showArrow>
            <IconButton
              aria-label={t('projects.openProject')}
              flexShrink={0}
              size="xs"
              variant="ghost"
              alignSelf="center"
              onClick={showOpenDialog}
            >
              <FolderOpenIcon />
            </IconButton>
          </Tooltip>
        </Tabs.List>
      </Tabs.Root>

      <ProjectTabContextMenu
        target={menuTarget}
        onClose={closeMenu}
        onCloseProject={closeProjectBySummary}
        onDelete={startDelete}
        onRename={startRename}
      />
      <RenameDialog
        initialName={renameTarget?.name ?? ''}
        isOpen={renameTarget !== null}
        onClose={closeRenameDialog}
        onSubmit={renameProject}
      />
      <ConfirmDialog
        body={t('projects.deleteProjectTabBody', { name: deleteTarget?.name ?? '' })}
        confirmLabel={t('projects.deleteProject')}
        isOpen={deleteTarget !== null}
        title={t('projects.deleteProjectQuestion')}
        onClose={closeDeleteDialog}
        onConfirm={confirmDeleteProject}
      />
      {isOpenDialogVisible ? <OpenProjectDialog isOpen onClose={hideOpenDialog} /> : null}
    </>
  );
};

const ProjectTab = ({
  backendConnectionStatus,
  isActive,
  loadingModelsCount,
  onCloseProject,
  onContextMenu,
  onSwitchProject,
  project,
}: {
  backendConnectionStatus: string;
  isActive: boolean;
  loadingModelsCount: number;
  onCloseProject: (project: ProjectTabSummary) => void;
  onContextMenu: (project: ProjectTabSummary, event: MouseEvent) => void;
  onSwitchProject: (projectId: string) => void;
  project: ProjectTabSummary;
}) => {
  const baseIndicatorState = getProjectQueueIndicatorState({
    isConnected: backendConnectionStatus === 'connected',
    loadingModelsCount,
    progress: null,
    queueItems: project.queueItems,
  });
  const progress = useQueueItemProgress(baseIndicatorState.runningQueueItemId ?? '');
  const { progressState } = getProjectQueueIndicatorState({
    isConnected: backendConnectionStatus === 'connected',
    loadingModelsCount,
    progress,
    queueItems: project.queueItems,
  });
  const switchProject = useCallback(() => onSwitchProject(project.id), [onSwitchProject, project.id]);
  const { t } = useTranslation();
  const openContextMenu = useCallback((event: MouseEvent) => onContextMenu(project, event), [onContextMenu, project]);

  const closeProject = useCallback(
    (event: MouseEvent) => {
      event.stopPropagation();
      onCloseProject(project);
    },
    [onCloseProject, project]
  );

  return (
    <Tabs.Trigger value={project.id} onClick={switchProject} onContextMenu={openContextMenu} fontSize="xs" h="full">
      {progressState.kind === 'idle' ? (
        <Icon as={isActive ? FolderOpenIcon : FolderIcon} boxSize="4" />
      ) : (
        <QueueCircularProgress size="2xs" state={progressState} />
      )}

      {project.name}

      <CloseButton
        size="2xs"
        me="-2"
        as="span"
        role="button"
        aria-label={t('projects.closeProjectLabel', { name: project.name })}
        onClick={closeProject}
      />
    </Tabs.Trigger>
  );
};

const ProjectTabContextMenu = ({
  onClose,
  onCloseProject,
  onDelete,
  onRename,
  target,
}: {
  onClose: () => void;
  onCloseProject: (project: ProjectTabSummary) => void;
  onDelete: (project: ProjectTabSummary) => void;
  onRename: (project: ProjectTabSummary) => void;
  target: { project: ProjectTabSummary; x: number; y: number } | null;
}) => {
  const dispatch = useWorkbenchDispatch();
  const store = useWorkbenchStore();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const targetRef = useRef(target);
  const { t } = useTranslation();

  targetRef.current = target;

  const positioning = useMemo(
    () => ({
      getAnchorRect: () => {
        const currentTarget = targetRef.current;

        return currentTarget ? { height: 1, width: 1, x: currentTarget.x, y: currentTarget.y } : null;
      },
      placement: 'bottom-start' as const,
    }),
    []
  );

  const openProjectDetails = useCallback(
    (projectSummary: ProjectTabSummary) => {
      const project = store.getState().projects.find((candidate) => candidate.id === projectSummary.id);

      if (!project) {
        return;
      }

      flushGenerateDrafts();
      dispatch({ projectId: project.id, type: 'switchProject' });

      // Reveal the Project panel wherever the project already shows it; default
      // to enabling it in the right rail. Both actions operate on the active
      // project, which the switch above just made this one.
      const { left, right } = project.widgetRegions;
      const region = right.instanceIds.some((instanceId) => project.widgetInstances[instanceId]?.typeId === 'project')
        ? 'right'
        : left.instanceIds.some((instanceId) => project.widgetInstances[instanceId]?.typeId === 'project')
          ? 'left'
          : null;
      const preferredRegions: WidgetRegion[] = [region ?? 'right'];

      openWorkbenchWidget('project', { preferredRegions });
    },
    [dispatch, openWorkbenchWidget, store]
  );

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  const renameTarget = useCallback(() => {
    if (targetRef.current) {
      onRename(targetRef.current.project);
    }
  }, [onRename]);

  const showDetails = useCallback(() => {
    if (targetRef.current) {
      openProjectDetails(targetRef.current.project);
    }
  }, [openProjectDetails]);

  const exportProject = useCallback(() => {
    const currentTarget = targetRef.current;
    const project = currentTarget
      ? store.getState().projects.find((candidate) => candidate.id === currentTarget.project.id)
      : null;

    if (project) {
      exportOpenProject(project);
    }
  }, [store]);

  const closeTargetProject = useCallback(() => {
    if (targetRef.current) {
      onCloseProject(targetRef.current.project);
    }
  }, [onCloseProject]);

  const deleteTargetProject = useCallback(() => {
    if (targetRef.current) {
      onDelete(targetRef.current.project);
    }
  }, [onDelete]);

  return (
    <Menu.Root
      key={target?.project.id ?? 'closed'}
      lazyMount
      open={target !== null}
      positioning={positioning}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Menu.Positioner>
          {target ? (
            <MenuContent minW="44">
              <Menu.Item value="rename" onClick={renameTarget}>
                <Icon as={PencilIcon} boxSize="3.5" />
                {t('projects.renameWithEllipsis')}
              </Menu.Item>
              <Menu.Item value="details" onClick={showDetails}>
                <Icon as={FolderCogIcon} boxSize="3.5" />
                {t('projects.projectDetails')}
              </Menu.Item>
              <Menu.Item value="export" onClick={exportProject}>
                <Icon as={FileDownIcon} boxSize="3.5" />
                {t('common.export')}
              </Menu.Item>
              <Menu.Separator />
              <Menu.Item value="close" onClick={closeTargetProject}>
                <Icon as={XIcon} boxSize="3.5" />
                {t('common.close')}
              </Menu.Item>
              <Menu.Separator />
              <Menu.Item color="fg.error" value="delete" _hover={deleteMenuItemHover} onClick={deleteTargetProject}>
                <Icon as={Trash2Icon} boxSize="3.5" />
                {t('projects.deleteProjectWithEllipsis')}
              </Menu.Item>
            </MenuContent>
          ) : null}
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
