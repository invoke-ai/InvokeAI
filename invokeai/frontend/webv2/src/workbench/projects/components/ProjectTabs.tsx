import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';

import { Flex, Icon, Menu, Portal, ScrollArea, Separator } from '@chakra-ui/react';
import { flushGenerateDrafts } from '@features/generation/drafts';
import { useModelLoads } from '@features/models';
import { getProjectQueueIndicatorState, type QueueItem } from '@features/queue/contracts';
import { useQueueItemProgress } from '@features/queue/react';
import { CloseButton, IconButton, ConfirmDialog, MenuContent, RenameDialog, Tabs, Tooltip } from '@platform/ui';
import { QueueCircularProgress } from '@workbench/components/QueueProgressIndicator';
import { exportOpenProject } from '@workbench/projects/projectFile';
import { useProjectActions } from '@workbench/projects/useProjectActions';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
  useWorkbenchSelector,
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
import { useCallback, useMemo, useState, type MouseEvent } from 'react';
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

// `min-width: 0` lets the strip shrink in flex layout, but does not stop its
// content from propagating into the tab list's intrinsic (shrink-to-fit)
// width — `contain: inline-size` zeroes that contribution so a long tab row
// can never stretch the list over the top bar's other controls. Same rule as
// the preview filmstrip.
const TAB_STRIP_CONTAIN_CSS = { contain: 'inline-size' } as const;

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
    (snapshot) => selectProjectTabSummaries(snapshot.projects),
    areProjectTabSummariesEqual
  );
  const activeProjectId = useActiveProjectSelector((project) => project.id);
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const modelLoads = useModelLoads();
  const queries = useWorkbenchQueries();
  const { projects } = useWorkbenchCommands();
  const { closeProject, deleteProject } = useProjectActions();
  const [menuTarget, setMenuTarget] = useState<{ project: ProjectTabSummary; x: number; y: number } | null>(null);
  const [renameTarget, setRenameTarget] = useState<ProjectTabSummary | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ProjectTabSummary | null>(null);
  const [isOpenDialogVisible, setIsOpenDialogVisible] = useState(false);

  const getProject = useCallback((projectId: string): Project | null => queries.getProject(projectId), [queries]);

  const onSwitchProject = useCallback(
    (projectId: string) => {
      flushGenerateDrafts();
      projects.switchTo(projectId);
    },
    [projects]
  );

  const openContextMenu = useCallback((project: ProjectTabSummary, event: MouseEvent) => {
    event.preventDefault();
    setMenuTarget({ project, x: event.clientX, y: event.clientY });
  }, []);

  const createProject = useCallback(() => {
    flushGenerateDrafts();
    projects.create();
  }, [projects]);

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
        projects.rename(renameTarget.id, name);
      }
    },
    [projects, renameTarget]
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
      <Tabs.Root minW="0" variant="subtle" value={activeProjectId} h="full" w="full">
        <Tabs.List flex="1 1 auto" minW="0" w="full" h="full" py="1">
          {/* Only the tabs scroll; the new/open controls stay pinned in view. */}
          <ScrollArea.Root css={TAB_STRIP_CONTAIN_CSS} flex="1 1 auto" minW="0" h="full" size="xs" variant="hover">
            <ScrollArea.Viewport aria-label={t('projects.openProjects')} h="full" w="full">
              <ScrollArea.Content asChild>
                <Flex align="center" h="full">
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
                </Flex>
              </ScrollArea.Content>
            </ScrollArea.Viewport>
            <ScrollArea.Scrollbar orientation="horizontal">
              <ScrollArea.Thumb />
            </ScrollArea.Scrollbar>
          </ScrollArea.Root>

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

  // Callback ref, not an effect: identity changes with `isActive`, so React
  // re-attaches it on every selection change and the newly active tab scrolls
  // into the strip's viewport.
  const scrollIntoViewWhenActive = useCallback(
    (node: HTMLButtonElement | null) => {
      if (node && isActive) {
        node.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
    },
    [isActive]
  );

  const closeProject = useCallback(
    (event: MouseEvent) => {
      event.stopPropagation();
      onCloseProject(project);
    },
    [onCloseProject, project]
  );

  return (
    <Tabs.Trigger
      ref={scrollIntoViewWhenActive}
      value={project.id}
      onClick={switchProject}
      onContextMenu={openContextMenu}
      fontSize="xs"
      flexShrink={0}
      h="full"
    >
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
  const { projects } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const { t } = useTranslation();

  const positioning = useMemo(
    () => ({
      getAnchorRect: () => {
        return target ? { height: 1, width: 1, x: target.x, y: target.y } : null;
      },
      placement: 'bottom-start' as const,
    }),
    [target]
  );

  const openProjectDetails = useCallback(
    (projectSummary: ProjectTabSummary) => {
      const project = queries.getProject(projectSummary.id);

      if (!project) {
        return;
      }

      flushGenerateDrafts();
      projects.switchTo(project.id);

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
    [openWorkbenchWidget, projects, queries]
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
    if (target) {
      onRename(target.project);
    }
  }, [onRename, target]);

  const showDetails = useCallback(() => {
    if (target) {
      openProjectDetails(target.project);
    }
  }, [openProjectDetails, target]);

  const exportProject = useCallback(() => {
    const project = target ? queries.getProject(target.project.id) : null;

    if (project) {
      exportOpenProject(project);
    }
  }, [queries, target]);

  const closeTargetProject = useCallback(() => {
    if (target) {
      onCloseProject(target.project);
    }
  }, [onCloseProject, target]);

  const deleteTargetProject = useCallback(() => {
    if (target) {
      onDelete(target.project);
    }
  }, [onDelete, target]);

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
