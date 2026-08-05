import type { Project } from '@workbench/projectContracts';
import type { ProjectSummary } from '@workbench/projects/library';

import { Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { flushGenerateDrafts } from '@features/generation/react';
import { useModelLoads } from '@features/models';
import { getProjectQueueIndicatorState } from '@features/queue/contracts';
import { useQueueItemProgress } from '@features/queue/react';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { MenuContent } from '@platform/ui/Menu';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { QueueCircularProgress } from '@workbench/components/QueueProgressIndicator';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { OpenProjectDialog } from '@workbench/projects/components';
import { refreshProjectLibrary, useProjectLibrarySelector } from '@workbench/projects/library';
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
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { projectSwitcherStore, setProjectSwitcherOpen } from './projectSwitcherStore';
import { HIDE_BELOW_PROJECT_NAME_WIDTH } from './topbarBreakpoints';

const MENU_POSITIONING = { placement: 'bottom-start' } as const;
const DELETE_HOVER_PROPS = { bg: 'bg.error', color: 'fg.error' } as const;
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

  const getProject = useCallback((projectId: string): Project | null => queries.getProject(projectId), [queries]);

  const createProject = useCallback(() => {
    flushGenerateDrafts();
    projects.create();
  }, [projects]);
  const showOpenDialog = useCallback(() => setIsOpenDialogVisible(true), []);
  const hideOpenDialog = useCallback(() => setIsOpenDialogVisible(false), []);
  const closeRenameDialog = useCallback(() => setRenameTarget(null), []);
  const closeDeleteDialog = useCallback(() => setDeleteTarget(null), []);
  const renameProject = useCallback(
    (name: string) => {
      if (renameTarget) {
        projects.rename(renameTarget.id, name);
      }
    },
    [projects, renameTarget]
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

    // Export bundles every image the project points at, so it runs for as long
    // as the project is large and can fail on a dropped connection or a project
    // past the archive ceiling. The reporter owns saying both.
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
        {/* No tooltip on a menu trigger: wrapping `Menu.Trigger` swallows the
            anchor ref, and the menu then renders in the page corner instead of
            under the control. The label already names it. */}
        <Menu.Trigger asChild>
          <Button
            aria-label={t('topbar.projectSwitcher.trigger', { name: activeProjectName })}
            size="sm"
            variant="ghost"
          >
            <Text css={HIDE_BELOW_PROJECT_NAME_WIDTH} fontWeight="500" minW="0" truncate>
              {activeProjectName}
            </Text>
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
                <Text fontSize="xs" fontWeight="700" truncate>
                  {activeProjectName}
                </Text>
              </Stack>
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
              <Menu.Item
                color="fg.error"
                value="delete-project"
                _hover={DELETE_HOVER_PROPS}
                onClick={deleteActiveProject}
              >
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

      {renameTarget ? (
        <RenameDialog initialName={renameTarget.name} isOpen onClose={closeRenameDialog} onSubmit={renameProject} />
      ) : null}
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
      <Menu.ItemText flex="1" minW="0" truncate>
        {project.name}
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
      <Menu.ItemText flex="1" minW="0" truncate>
        {summary.name}
      </Menu.ItemText>
      <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
        {t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
      </Text>
    </Menu.Item>
  );
};
