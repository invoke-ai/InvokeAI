/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop -- List slots take JSX and callbacks; the React Compiler memoizes them. */
import type { ListRowProps } from '@platform/ui/list/List';
import type { ProjectWithWorkspaceDTO } from '@workbench/projects/api';

import { Box, Icon, Portal, Spinner, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { areArraysEqual } from '@platform/state/selectors';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { getListContentHeight, List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { listRowsFromItems } from '@platform/ui/list/listRows';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { ProjectCompatibilityBadge } from '@workbench/launchpad/projects/ProjectCompatibilityBadge';
import { ProjectsLoadError } from '@workbench/launchpad/projects/ProjectsLoadError';
import {
  isProjectSummaryCompatible,
  refreshProjectLibrary,
  useProjectLibrarySelector,
  type ProjectSummary,
} from '@workbench/projects/library';
import { describeRefusedProject } from '@workbench/projects/projectLoadRefusal';
import { useImportProjectFile } from '@workbench/projects/useProjectFileActions';
import { useNotify } from '@workbench/useNotify';
import {
  useWorkbenchCommands,
  useWorkbenchPersistenceService,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { ArrowRightIcon, FileUpIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Title, edited time and an inline compatibility badge fit the comfortable pitch. */
const LIST_DENSITY = 'comfortable';
/** Five rows and a bit, so a longer library visibly scrolls. */
const MAX_LIST_HEIGHT_PX = 288;

const getSummaryKey = (summary: ProjectSummary) => summary.id;

const ProjectLibraryRefresh = () => {
  useMountEffect(() => {
    void refreshProjectLibrary();
  });

  return null;
};

/** Exclude open tabs; hydrate in place to avoid remounting the editor. */
export const OpenProjectDialog = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const projectIds = useWorkbenchSelector((snapshot) => snapshot.projects.map((project) => project.id), areArraysEqual);
  const { projects } = useWorkbenchCommands();
  const persistence = useWorkbenchPersistenceService();
  const notify = useNotify();
  const { t } = useTranslation();
  const summaries = useProjectLibrarySelector((snapshot) => snapshot.summaries);
  const libraryStatus = useProjectLibrarySelector((snapshot) => snapshot.status);
  const [busyProjectId, setBusyProjectId] = useState<string | null>(null);

  const openProjectIds = useMemo(() => new Set(projectIds), [projectIds]);
  const available = useMemo(
    () => summaries.filter((summary) => !openProjectIds.has(summary.id)),
    [openProjectIds, summaries]
  );
  const rows = useMemo(() => listRowsFromItems(available, getSummaryKey), [available]);
  // Keyboard users start on the first project they can open; incompatible ones stay reachable with the arrows.
  const firstOpenableKey = available.find(isProjectSummaryCompatible)?.id ?? null;
  const listRef = useRef<HTMLDivElement>(null);
  const importRef = useRef<HTMLButtonElement>(null);
  // After a retry recovers: the list's tab stop, or Import when nothing came back to open.
  const getRecoveryFocus = useCallback(
    () => listRef.current?.querySelector<HTMLElement>('button[tabindex="0"]') ?? importRef.current,
    []
  );
  // A failed listing with nothing listed earlier is not an empty library; earlier rows stay usable after a failure.
  const listStatus =
    summaries.length > 0
      ? 'ready'
      : libraryStatus === 'error'
        ? 'error'
        : libraryStatus === 'ready'
          ? 'ready'
          : 'loading';

  const openProject = useCallback(
    async (summary: ProjectSummary) => {
      const owner = captureAccountScope();

      setBusyProjectId(summary.id);

      try {
        const result = await persistence.hydrateProjectFromServer(summary.id, summary.name);

        assertAccountScopeCurrent(owner);
        if (result.status === 'refused') {
          const notice = describeRefusedProject(result.refused, t);

          notify.error(notice.title, notice.message);

          return;
        }
        if (result.status !== 'loaded') {
          notify.error(t('projects.couldNotOpen'), t('projects.couldNotOpenDescription', { name: summary.name }));
          void refreshProjectLibrary();

          return;
        }

        projects.open(result.project);
        onClose();
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        notify.error(
          t('projects.couldNotOpen'),
          getApiErrorMessage(error, t('projects.couldNotOpenDescription', { name: summary.name }))
        );
      } finally {
        if (isAccountScopeCurrent(owner)) {
          setBusyProjectId(null);
        }
      }
    },
    [notify, onClose, persistence, projects, t]
  );

  const openImportedProject = useCallback(
    (record: ProjectWithWorkspaceDTO) => {
      const result = persistence.adoptProjectRecord(record);

      if (result.status === 'loaded') {
        projects.open(result.project);
        onClose();
      } else if (result.status === 'refused') {
        const notice = describeRefusedProject(result.refused, t);

        notify.error(notice.title, notice.message);
      }
    },
    [notify, onClose, persistence, projects, t]
  );
  const handleImport = useImportProjectFile(openImportedProject);

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  const startImport = useCallback(() => void handleImport(), [handleImport]);

  // An incompatible project stays listed, dimmed, with its badge saying why it cannot open here.
  const renderItem = (summary: ProjectSummary, rowProps: ListRowProps) => {
    const isCompatible = isProjectSummaryCompatible(summary);

    return (
      <ListItem
        {...rowProps}
        badges={isCompatible ? undefined : <ProjectCompatibilityBadge summary={summary} />}
        description={t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
        isDisabled={!isCompatible}
        title={summary.name}
        trailing={
          busyProjectId === summary.id ? (
            <Spinner color="fg.muted" />
          ) : isCompatible ? (
            <Icon as={ArrowRightIcon} boxSize="3.5" />
          ) : undefined
        }
        onPress={() => void openProject(summary)}
      />
    );
  };

  return (
    <Dialog.Root lazyMount open={isOpen} size="sm" unmountOnExit onOpenChange={handleOpenChange}>
      {isOpen ? <ProjectLibraryRefresh /> : null}
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>{t('projects.openProject')}</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Box
                ref={listRef}
                h={
                  rows.length > 0
                    ? `${Math.min(getListContentHeight(rows, LIST_DENSITY), MAX_LIST_HEIGHT_PX)}px`
                    : undefined
                }
                // Row surfaces bleed into the dialog padding so their text lines up with the title.
                mx="-2"
              >
                <List
                  defaultFocusKey={firstOpenableKey}
                  density={LIST_DENSITY}
                  dividers
                  inset="none"
                  emptyState={
                    <Text color="fg.muted" fontSize="md" py="2" textAlign="center">
                      {summaries.length === 0 ? t('projects.noSavedProjects') : t('projects.allSavedAlreadyOpen')}
                    </Text>
                  }
                  errorState={<ProjectsLoadError getRecoveryFocus={getRecoveryFocus} layout="state" />}
                  isBusy={busyProjectId !== null}
                  label={t('projects.openProject')}
                  renderItem={renderItem}
                  rows={rows}
                  status={listStatus}
                />
              </Box>
            </Dialog.Body>
            <Dialog.Footer justifyContent="space-between">
              <Button ref={importRef} variant="outline" onClick={startImport}>
                <FileUpIcon />
                {t('projects.importWithEllipsis')}
              </Button>
              <Button variant="ghost" onClick={onClose}>
                {t('common.cancel')}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
