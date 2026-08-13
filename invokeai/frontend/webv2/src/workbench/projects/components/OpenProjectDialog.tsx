import type { ProjectRecordDTO } from '@workbench/projects/api';

import { Dialog, Icon, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { flushGenerateDrafts } from '@features/generation/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { areArraysEqual } from '@platform/state/selectors';
import { Button, CloseButton, Row, Scrollable } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { refreshProjectLibrary, useProjectLibrarySelector, type ProjectSummary } from '@workbench/projects/library';
import { useImportProjectFile } from '@workbench/projects/useProjectFileActions';
import { useNotify } from '@workbench/useNotify';
import {
  useWorkbenchCommands,
  useWorkbenchPersistenceService,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { ArrowRightIcon, FileUpIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const disabledRowStyles = { opacity: 0.6 } as const;

const ProjectLibraryRefresh = () => {
  useMountEffect(() => {
    void refreshProjectLibrary();
  });

  return null;
};

/**
 * "Open project…" from the tab bar: the saved projects that are not already
 * open as tabs, plus import. Selecting one hydrates its document from the
 * library and opens it in place — no navigation, the editor stays mounted.
 */
export const OpenProjectDialog = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const projectIds = useWorkbenchSelector((snapshot) => snapshot.projects.map((project) => project.id), areArraysEqual);
  const { projects } = useWorkbenchCommands();
  const persistence = useWorkbenchPersistenceService();
  const notify = useNotify();
  const { t } = useTranslation();
  const summaries = useProjectLibrarySelector((snapshot) => snapshot.summaries);
  const [busyProjectId, setBusyProjectId] = useState<string | null>(null);

  const openProjectIds = useMemo(() => new Set(projectIds), [projectIds]);
  const available = useMemo(
    () => summaries.filter((summary) => !openProjectIds.has(summary.id)),
    [openProjectIds, summaries]
  );

  const openProject = useCallback(
    async (summary: ProjectSummary) => {
      const owner = captureAccountScope();

      setBusyProjectId(summary.id);

      try {
        const project = await persistence.hydrateProjectFromServer(summary.id);

        assertAccountScopeCurrent(owner);
        if (!project) {
          notify.error(t('projects.couldNotOpen'), t('projects.couldNotOpenDescription', { name: summary.name }));
          void refreshProjectLibrary();

          return;
        }

        flushGenerateDrafts();
        projects.open(project);
        onClose();
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        notify.error(
          t('projects.couldNotOpen'),
          error instanceof Error ? error.message : t('projects.couldNotOpenDescription', { name: summary.name })
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
    (record: ProjectRecordDTO) => {
      const project = persistence.adoptProjectRecord(record);

      if (project) {
        flushGenerateDrafts();
        projects.open(project);
        onClose();
      }
    },
    [onClose, persistence, projects]
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

  return (
    <Dialog.Root lazyMount open={isOpen} placement="center" size="sm" unmountOnExit onOpenChange={handleOpenChange}>
      {isOpen ? <ProjectLibraryRefresh /> : null}
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>{t('projects.openProject')}</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Scrollable maxH="72">
                <Stack gap="1">
                  {available.map((summary) => (
                    <OpenProjectRow
                      key={summary.id}
                      isBusy={busyProjectId === summary.id}
                      isDisabled={busyProjectId !== null}
                      summary={summary}
                      onOpen={openProject}
                    />
                  ))}
                  {available.length === 0 ? (
                    <Text color="fg.muted" fontSize="xs" px="2.5" py="4" textAlign="center">
                      {summaries.length === 0 ? t('projects.noSavedProjects') : t('projects.allSavedAlreadyOpen')}
                    </Text>
                  ) : null}
                </Stack>
              </Scrollable>
            </Dialog.Body>
            <Dialog.Footer gap="2" justifyContent="space-between">
              <Button size="xs" variant="outline" onClick={startImport}>
                <FileUpIcon />
                {t('projects.importWithEllipsis')}
              </Button>
              <Button size="xs" variant="ghost" onClick={onClose}>
                {t('common.cancel')}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton color="fg.muted" size="sm" />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

const OpenProjectRow = ({
  isBusy,
  isDisabled,
  onOpen,
  summary,
}: {
  isBusy: boolean;
  isDisabled: boolean;
  onOpen: (summary: ProjectSummary) => Promise<void>;
  summary: ProjectSummary;
}) => {
  const open = useCallback(() => void onOpen(summary), [onOpen, summary]);
  const { t } = useTranslation();

  return (
    <Row asChild gap="2.5" px="2.5" py="2" rounded="md" _disabled={disabledRowStyles}>
      <button disabled={isDisabled} type="button" onClick={open}>
        <Stack flex="1" gap="0" minW="0">
          <MiddleTruncate fontSize="xs" fontWeight="600" text={summary.name} />
          <Text color="fg.muted" fontSize="2xs">
            {t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
          </Text>
        </Stack>
        {isBusy ? <Spinner color="fg.muted" size="xs" /> : <Icon as={ArrowRightIcon} boxSize="3.5" color="fg.muted" />}
      </button>
    </Row>
  );
};
