import type { Project } from '@workbench/projectContracts';
import type { ProjectSummary } from '@workbench/projects/library';
import type { MouseEvent } from 'react';

import { Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { flushGenerateDrafts } from '@features/generation/react';
import { useModelLoads } from '@features/models';
import { getProjectQueueIndicatorState } from '@features/queue/contracts';
import { useQueueItemProgress } from '@features/queue/react';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { MenuContent } from '@platform/ui/Menu';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { toaster } from '@platform/ui/toaster';
import { QueueCircularProgress } from '@workbench/components/QueueProgressIndicator';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { OpenProjectDialog } from '@workbench/projects/components';
import { refreshProjectLibrary, renameLibraryProject, useProjectLibrarySelector } from '@workbench/projects/library';
import { useProjectActions } from '@workbench/projects/useProjectActions';
import { useExportOpenProject } from '@workbench/projects/useProjectFileActions';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import {
  CheckIcon,
  ChevronsUpDownIcon,
  FileDownIcon,
  FolderCogIcon,
  FolderOpenIcon,
  PencilIcon,
  PlusIcon,
  Trash2Icon,
  XIcon,
} from 'lucide-react';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { projectSwitcherStore, setProjectSwitcherOpen } from './projectSwitcherStore';
import { HIDE_BELOW_PROJECT_NAME_WIDTH } from './topbarBreakpoints';

const MENU_POSITIONING = { placement: 'bottom-start' } as const;
const RECENT_PROJECT_LIMIT = 5;

export const ProjectSwitcher = () => {
  const { t } = useTranslation();
  const activeProjectId = useActiveProjectSelector((project) => project.id);
  const activeProjectName = useActiveProjectSelector((project) => project.name);
  const librarySummaries = useProjectLibrarySelector((snapshot) => snapshot.summaries);
  const openProjects = useWorkbenchSelector((snapshot) => snapshot.projects);
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const modelLoads = useModelLoads();
  const queries = useWorkbenchQueries();
  const { projects } = useWorkbenchCommands();
  const { closeProject, deleteProject, openProject } = useProjectActions();
  const startExport = useExportOpenProject();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const returnFocus = useCallback(() => triggerRef.current, []);
  const [renameTarget, setRenameTarget] = useState<{ id: string; name: string } | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null);
  const [isOpenDialogVisible, setIsOpenDialogVisible] = useState(false);

  const isMenuOpen = projectSwitcherStore.useSelector((snapshot) => snapshot.isOpen);
  const handleMenuOpenChange = useCallback((event: { open: boolean }) => {
    setProjectSwitcherOpen(event.open);

    if (event.open) {
      void refreshProjectLibrary();
    }
  }, []);

  // Right-click opens the same dropdown: the trigger IS this control's context.
  const handleTriggerContextMenu = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      handleMenuOpenChange({ open: true });
    },
    [handleMenuOpenChange]
  );

  const getProject = useCallback((projectId: string): Project | null => queries.getProject(projectId), [queries]);

  const createProject = useCallback(() => {
    flushGenerateDrafts();
    projects.create();
  }, [projects]);
  const showOpenDialog = useCallback(() => setIsOpenDialogVisible(true), []);
  const hideOpenDialog = useCallback(() => setIsOpenDialogVisible(false), []);
  const closeRenameDialog = useCallback(() => setRenameTarget(null), []);
  const closeDeleteDialog = useCallback(() => setDeleteTarget(null), []);
  // Through the library, not the bare store command: the library path flushes
  // an open project immediately, which is what renames its board on the server.
  const renameProject = useCallback(
    async (name: string) => {
      if (!renameTarget) {
        return;
      }

      try {
        await renameLibraryProject(renameTarget.id, name);
      } catch (error) {
        toaster.create({
          description: error instanceof Error ? error.message : undefined,
          title: t('projects.renameFailed'),
          type: 'error',
        });
        // The rename dialog stays open on a rejection, so the user keeps their input.
        throw error;
      }
    },
    [renameTarget, t]
  );
  const confirmDeleteProject = useCallback(async () => {
    const project = deleteTarget ? getProject(deleteTarget.id) : null;

    if (project) {
      await deleteProject(project);
    }
  }, [deleteProject, deleteTarget, getProject]);
  const renameActiveProject = useCallback(
    () => setRenameTarget({ id: activeProjectId, name: activeProjectName }),
    [activeProjectId, activeProjectName]
  );
  const deleteActiveProject = useCallback(
    () => setDeleteTarget({ id: activeProjectId, name: activeProjectName }),
    [activeProjectId, activeProjectName]
  );
  const showActiveProjectDetails = useCallback(() => openWorkbenchWidget('project'), [openWorkbenchWidget]);
  const exportActiveProject = useCallback(() => {
    flushGenerateDrafts();

    const project = getProject(activeProjectId);

    if (!project) {
      return;
    }

    // Use the reporter for export progress and failures, including connection loss and archive limits.
    startExport(project);
  }, [activeProjectId, getProject, startExport]);
  const closeActiveProject = useCallback(() => {
    const project = getProject(activeProjectId);

    if (project) {
      closeProject(project);
    }
  }, [activeProjectId, closeProject, getProject]);
  const selectOpenProject = useCallback(
    (event: { value: string }) => {
      const project = getProject(event.value);

      if (project) {
        void openProject(project.id, project.name);
      }
    },
    [getProject, openProject]
  );

  const openProjectIds = new Set(openProjects.map((project) => project.id));
  const openProjectSummaries = openProjects;
  const recentSummaries = librarySummaries
    .filter((project) => !openProjectIds.has(project.id))
    .slice(0, RECENT_PROJECT_LIMIT);

  return (
    <>
      <Menu.Root open={isMenuOpen} positioning={MENU_POSITIONING} onOpenChange={handleMenuOpenChange}>
        {/* This labeled trigger supplies its own name; wrapping it with Tooltip would lose the menu anchor ref. */}
        <Menu.Trigger asChild>
          <Button
            ref={triggerRef}
            aria-label={t('topbar.projectSwitcher.trigger', { name: activeProjectName })}
            size="sm"
            variant="ghost"
            onContextMenu={handleTriggerContextMenu}
          >
            <MiddleTruncate css={HIDE_BELOW_PROJECT_NAME_WIDTH} fontWeight="500" minW="0" text={activeProjectName} />
            <Icon as={ChevronsUpDownIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent maxW="22rem" minW="18rem">
              <Stack gap="0" px="3" py="2">
                <Text color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  {t('projects.projectDetails')}
                </Text>
                <MiddleTruncate fontSize="xs" fontWeight="700" text={activeProjectName} />
              </Stack>
              <Menu.Separator />
              <Menu.Item value="rename-project" onClick={renameActiveProject}>
                <Icon as={PencilIcon} boxSize="3.5" />
                <Menu.ItemText>{t('projects.renameWithEllipsis')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="project-details" onClick={showActiveProjectDetails}>
                <Icon as={FolderCogIcon} boxSize="3.5" />
                <Menu.ItemText>{t('common.details')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="export-project" onClick={exportActiveProject}>
                <Icon as={FileDownIcon} boxSize="3.5" />
                <Menu.ItemText>{t('common.export')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="close-project" onClick={closeActiveProject}>
                <Icon as={XIcon} boxSize="3.5" />
                <Menu.ItemText>{t('common.close')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item data-danger="" value="delete-project" onClick={deleteActiveProject}>
                <Icon as={Trash2Icon} boxSize="3.5" />
                <Menu.ItemText>{t('projects.deleteProjectWithEllipsis')}</Menu.ItemText>
              </Menu.Item>

              <Menu.Separator />
              <Menu.RadioItemGroup value={activeProjectId} onValueChange={selectOpenProject}>
                <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  {t('projects.openProjects')}
                </Menu.ItemGroupLabel>
                {openProjectSummaries.map((project) => (
                  <OpenProjectRow
                    key={project.id}
                    backendConnectionStatus={backendConnectionStatus}
                    loadingModelsCount={modelLoads.length}
                    project={project}
                  />
                ))}
              </Menu.RadioItemGroup>

              {recentSummaries.length > 0 ? (
                <>
                  <Menu.Separator />
                  <Menu.ItemGroup>
                    <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                      {t('common.recent')}
                    </Menu.ItemGroupLabel>
                    {recentSummaries.map((summary) => (
                      <RecentProjectRow key={summary.id} summary={summary} onOpen={openProject} />
                    ))}
                  </Menu.ItemGroup>
                </>
              ) : null}

              <Menu.Separator />
              <Menu.Item value="new-project" onClick={createProject}>
                <Icon as={PlusIcon} boxSize="3.5" />
                <Menu.ItemText>{t('projects.newProject')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="open-project" onClick={showOpenDialog}>
                <Icon as={FolderOpenIcon} boxSize="3.5" />
                <Menu.ItemText>{t('projects.openProject')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <RenameDialog
        finalFocusEl={returnFocus}
        initialName={renameTarget?.name ?? ''}
        isOpen={renameTarget !== null}
        label={t('projects.renameProjectNameLabel')}
        submitLabel={t('common.rename')}
        title={t('projects.renameProject')}
        onClose={closeRenameDialog}
        onSubmit={renameProject}
      />
      <ConfirmDialog
        body={`${t('projects.deleteProjectTabBody', { name: deleteTarget?.name ?? '' })} ${t('projects.deleteProjectBoardNote')}`}
        confirmLabel={t('projects.deleteProject')}
        isOpen={deleteTarget !== null}
        title={t('projects.deleteProjectQuestion')}
        onClose={closeDeleteDialog}
        onConfirm={confirmDeleteProject}
      />
      {/*
       * Keep the dialog mounted by open state; its lazyMount/unmountOnExit handle closed cost and preserve exit
       * animation.
       */}
      <OpenProjectDialog isOpen={isOpenDialogVisible} onClose={hideOpenDialog} />
    </>
  );
};

const OpenProjectRow = ({
  backendConnectionStatus,
  loadingModelsCount,
  project,
}: {
  backendConnectionStatus: string;
  loadingModelsCount: number;
  project: Project;
}) => {
  const baseState = getProjectQueueIndicatorState({
    isConnected: backendConnectionStatus === 'connected',
    loadingModelsCount,
    progress: null,
    queueItems: project.queue.items,
  });
  const progress = useQueueItemProgress(baseState.runningQueueItemId ?? '');
  const { progressState } = getProjectQueueIndicatorState({
    isConnected: backendConnectionStatus === 'connected',
    loadingModelsCount,
    progress,
    queueItems: project.queue.items,
  });
  return (
    <Menu.RadioItem value={project.id}>
      <Menu.ItemIndicator color="accent.fg">
        <Icon as={CheckIcon} boxSize="3.5" />
      </Menu.ItemIndicator>
      <Menu.ItemText flex="1" minW="0">
        <MiddleTruncate as="span" text={project.name} />
      </Menu.ItemText>
      <QueueCircularProgress state={progressState} />
    </Menu.RadioItem>
  );
};

const RecentProjectRow = ({
  onOpen,
  summary,
}: {
  summary: ProjectSummary;
  onOpen: (projectId: string, name: string) => Promise<void>;
}) => {
  const { t } = useTranslation();
  const handleSelect = useCallback(() => void onOpen(summary.id, summary.name), [onOpen, summary]);

  return (
    <Menu.Item value={summary.id} onClick={handleSelect}>
      <Menu.ItemText flex="1" minW="0">
        <MiddleTruncate as="span" text={summary.name} />
      </Menu.ItemText>
      <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
        {t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
      </Text>
    </Menu.Item>
  );
};
