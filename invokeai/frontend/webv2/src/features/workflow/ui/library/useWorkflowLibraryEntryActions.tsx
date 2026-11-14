import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';

import { getWorkflowRecordCategory } from '@features/workflow/data/api';
import {
  createLibraryWorkflow,
  deleteLibraryWorkflow,
  getLibraryWorkflowCached,
  getLibraryWorkflowRecord,
  getLibraryWorkflowRecordCached,
  invalidateWorkflowLibraryCache,
  updateLibraryWorkflow,
} from '@features/workflow/queries';
import { useWorkflowGraphPreview, useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import { parseWorkflowJson } from '@features/workflow/utility';
import { downloadText } from '@platform/browser/downloadBlob';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { MenuActionItem } from '@platform/ui/Menu';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { CopyIcon, DownloadIcon, GitForkIcon, PencilIcon, Trash2Icon, WorkflowIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { findProjectCopiesOf } from './projectWorkflowEntries';

/**
 * Workflows whose duplicate is in flight. One guard serves every surface that offers Duplicate (the rail and a card's
 * menu), so one workflow gets one copy per press, and a pending copy of one workflow leaves the others free.
 */
export interface WorkflowLibraryDuplicateGuard {
  pendingIds: ReadonlySet<string>;
  /** False when that workflow's duplicate is already in flight. */
  begin: (workflowId: string) => boolean;
  end: (workflowId: string) => void;
}

/** Owned by the surface that hosts every Duplicate of a view: the template panes. */
export const useWorkflowLibraryDuplicateGuard = (): WorkflowLibraryDuplicateGuard => {
  const [pendingIds, setPendingIds] = useState<ReadonlySet<string>>(() => new Set());
  // The ref answers a second press before the first press's render lands.
  const pendingRef = useRef(new Set<string>());
  const begin = useCallback((workflowId: string) => {
    if (pendingRef.current.has(workflowId)) {
      return false;
    }

    pendingRef.current.add(workflowId);
    setPendingIds(new Set(pendingRef.current));
    return true;
  }, []);
  const end = useCallback((workflowId: string) => {
    pendingRef.current.delete(workflowId);
    setPendingIds(new Set(pendingRef.current));
  }, []);

  return useMemo(() => ({ begin, end, pendingIds }), [begin, end, pendingIds]);
};

export interface WorkflowLibraryEntryActionsOptions {
  duplicateGuard: WorkflowLibraryDuplicateGuard;
  /** The workflow the actions act on: the rail's selection, or the card a context menu was raised on. */
  entry: WorkflowLibraryEntry | null;
  /** Whether the account may not rename or delete `entry` (`getLibraryEntryAccess`); the server still enforces. */
  isReadOnly: boolean;
  /** The project's workflows, so Open can say it will ask about existing copies. */
  projectWorkflows: readonly ProjectWorkflowEntry[];
  /** The shell closes the library when a fork takes the user to a new project. */
  onClose: () => void;
  /** Receives the deleted workflow's id. */
  onDeleted: (workflowId: string) => void;
  /** Carries the copy's id so the shell can select it once the list refreshes. */
  onDuplicated: (workflowId: string) => void;
  /** Adds the first copy, or asks what to do when the project already holds one. */
  onOpen: (item: WorkflowLibraryListItem) => void;
}

export interface WorkflowLibraryEntryActions {
  /** The menu items: Open, Duplicate, Fork, Download JSON and, for the account's own templates, Rename and Delete. */
  actionItems: ReactNode;
  /** The rename and delete dialogs the items open; render them outside anything a delete empties. */
  dialogs: ReactNode;
  open: () => void;
  /** "Open…" once the project holds a copy, since opening then asks what to do (`planLibraryWorkflowOpen`). */
  openLabel: string;
  projectCopies: readonly ProjectWorkflowEntry[];
}

/**
 * The workflow a dialog was opened for, and its name then. Write access was checked when it opened; the dialog
 * answers to this target alone, so a refresh that drops the entry neither hides it nor leaves it to reappear later.
 */
interface DialogTarget {
  name: string;
  workflowId: string;
}

const toFileSlug = (name: string): string => name.trim().replaceAll(/\s+/g, '-').toLowerCase() || 'workflow';

/**
 * One library workflow's actions and the dialogs they open. Every action reads the record by id and revision, so it
 * acts on `entry` whether or not it is the selection; each caller (the detail rail, a card's context menu) owns its
 * own instance, and with it its own rename and delete dialogs.
 */
export const useWorkflowLibraryEntryActions = ({
  duplicateGuard,
  entry,
  isReadOnly,
  projectWorkflows,
  onClose,
  onDeleted,
  onDuplicated,
  onOpen,
}: WorkflowLibraryEntryActionsOptions): WorkflowLibraryEntryActions => {
  const { t } = useTranslation();
  const notify = useWorkflowNotifications();
  const { openDocumentInNewProject } = useWorkflowGraphPreview();
  // Captured on request: a delete moves the selection, or empties the rail, before its dialog closes, and a list
  // refresh can drop a context menu's target while a dialog it opened is still up.
  const [deleteTarget, setDeleteTarget] = useState<DialogTarget | null>(null);
  const [renameTarget, setRenameTarget] = useState<DialogTarget | null>(null);

  const item = entry?.item ?? null;
  const projectCopies = useMemo(
    () => (item ? findProjectCopiesOf(projectWorkflows, item.workflow_id) : []),
    [item, projectWorkflows]
  );
  const hasCopies = projectCopies.length > 0;
  const openLabel = hasCopies ? t('workflowLibrary.openWithEllipsis') : t('workflowLibrary.open');

  const open = useCallback(() => {
    if (item) {
      onOpen(item);
    }
  }, [item, onOpen]);

  /** Duplicate the library record directly without loading or modifying the active project graph. */
  const duplicate = useCallback(async () => {
    // Guard until the copy exists, including copies landing outside the visible category.
    if (!item || !duplicateGuard.begin(item.workflow_id)) {
      return;
    }

    const owner = captureAccountScope();

    try {
      const raw = await getLibraryWorkflowCached(item.workflow_id, {
        expectedRevision: item.revision,
        signal: owner.signal,
      });

      assertAccountScopeCurrent(owner);

      const { id: _id, ...copy } = raw;
      const meta = typeof copy.meta === 'object' && copy.meta !== null ? (copy.meta as Record<string, unknown>) : {};
      const workflowId = await createLibraryWorkflow(
        {
          ...copy,
          // A copy of a default is still the user's own workflow.
          meta: { ...meta, category: 'user' },
          name: t('workflowLibrary.duplicateName', { name: item.name || t('workflowLibrary.untitled') }),
        },
        owner.signal
      );

      assertAccountScopeCurrent(owner);
      invalidateWorkflowLibraryCache();
      // The copy is forced into the user category, so from the Browse tab it
      // lands out of sight — the notice is the only thing that says it worked.
      notify.success(t('workflowLibrary.duplicated'));
      onDuplicated(workflowId);
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.duplicateFailed'), getApiErrorMessage(error, t('common.unknownError')));
      }
    } finally {
      duplicateGuard.end(item.workflow_id);
    }
  }, [duplicateGuard, item, notify, onDuplicated, t]);
  const handleDuplicate = useCallback(() => void duplicate(), [duplicate]);

  const openRename = useCallback(() => {
    if (item && !isReadOnly) {
      setRenameTarget({ name: item.name || t('workflowLibrary.untitled'), workflowId: item.workflow_id });
    }
  }, [isReadOnly, item, t]);
  const closeRename = useCallback(() => setRenameTarget(null), []);
  // A rename is a content write at the record's current revision: the live record, never a cached one, is the base.
  const submitRename = useCallback(
    async (nextName: string) => {
      if (!renameTarget) {
        return;
      }

      const owner = captureAccountScope();
      const { workflowId } = renameTarget;

      try {
        const record = await getLibraryWorkflowRecord(workflowId, owner.signal);

        assertAccountScopeCurrent(owner);
        await updateLibraryWorkflow(
          workflowId,
          { ...record.workflow, name: nextName },
          { expectedRevision: record.revision, signal: owner.signal }
        );
        assertAccountScopeCurrent(owner);
        invalidateWorkflowLibraryCache(workflowId);
        notify.success(t('workflowLibrary.renamed'));
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        notify.error(t('workflowLibrary.renameFailed'), getApiErrorMessage(error, t('common.unknownError')));
        // Rejecting keeps the dialog, and the typed name, open for another try.
        throw error;
      }
    },
    [notify, renameTarget, t]
  );

  const fork = useCallback(async () => {
    if (!item) {
      return;
    }

    const owner = captureAccountScope();

    try {
      const record = await getLibraryWorkflowRecordCached(item.workflow_id, {
        expectedRevision: item.revision,
        signal: owner.signal,
      });

      assertAccountScopeCurrent(owner);

      const { document } = parseWorkflowJson({ ...record.workflow, id: record.workflow_id });

      // The port creates and activates a fresh project first, so the project
      // the library was opened from is left exactly as it was.
      openDocumentInNewProject(document, item.name, {
        libraryWorkflowId: record.workflow_id,
        ...(getWorkflowRecordCategory(record) === 'user' ? { ownerUserId: record.user_id } : {}),
        revision: record.revision,
      });
      onClose();
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.loadFailed'), getApiErrorMessage(error, t('common.unknownError')));
      }
    }
  }, [item, notify, onClose, openDocumentInNewProject, t]);
  const handleFork = useCallback(() => void fork(), [fork]);

  const download = useCallback(async () => {
    if (!item) {
      return;
    }

    const owner = captureAccountScope();

    try {
      const raw = await getLibraryWorkflowCached(item.workflow_id, {
        expectedRevision: item.revision,
        signal: owner.signal,
      });

      assertAccountScopeCurrent(owner);
      downloadText(JSON.stringify(raw, null, 2), `${toFileSlug(item.name)}.json`, 'application/json');
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.loadFailed'), getApiErrorMessage(error, t('common.unknownError')));
      }
    }
  }, [item, notify, t]);
  const handleDownload = useCallback(() => void download(), [download]);

  const openDeleteConfirm = useCallback(() => {
    if (item && !isReadOnly) {
      setDeleteTarget({ name: item.name || t('workflowLibrary.untitled'), workflowId: item.workflow_id });
    }
  }, [isReadOnly, item, t]);
  const closeDeleteConfirm = useCallback(() => setDeleteTarget(null), []);
  const confirmDelete = useCallback(async () => {
    if (!deleteTarget) {
      return;
    }

    const owner = captureAccountScope();
    const { workflowId } = deleteTarget;

    try {
      await deleteLibraryWorkflow(workflowId, owner.signal);

      assertAccountScopeCurrent(owner);
      invalidateWorkflowLibraryCache();
      onDeleted(workflowId);
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.deleteFailed'), getApiErrorMessage(error, t('common.unknownError')));
      }
    }
  }, [deleteTarget, notify, onDeleted, t]);

  // Bundled defaults and other accounts' templates are not the account's to rename or delete.
  const canWrite = item !== null && !isReadOnly;
  const actionItems = (
    <>
      <MenuActionItem icon={WorkflowIcon} label={openLabel} value="open" onSelect={open} />
      <MenuActionItem
        icon={CopyIcon}
        disabled={item !== null && duplicateGuard.pendingIds.has(item.workflow_id)}
        label={t('workflowLibrary.duplicate')}
        value="duplicate"
        onSelect={handleDuplicate}
      />
      <MenuActionItem
        icon={GitForkIcon}
        label={t('workflowLibrary.forkIntoProject')}
        value="fork-into-project"
        onSelect={handleFork}
      />
      <MenuActionItem
        icon={DownloadIcon}
        label={t('workflowLibrary.downloadJson')}
        value="download-json"
        onSelect={handleDownload}
      />
      {canWrite ? (
        <MenuActionItem
          icon={PencilIcon}
          label={t('workflowLibrary.renameWithEllipsis')}
          value="rename"
          onSelect={openRename}
        />
      ) : null}
      {canWrite ? (
        <MenuActionItem
          icon={Trash2Icon}
          label={t('workflowLibrary.delete')}
          tone="danger"
          value="delete"
          onSelect={openDeleteConfirm}
        />
      ) : null}
    </>
  );

  const dialogs = (
    <>
      <RenameDialog
        initialName={renameTarget?.name ?? ''}
        isOpen={renameTarget !== null}
        label={t('workflowLibrary.templateName')}
        submitLabel={t('workflowLibrary.rename')}
        title={t('workflowLibrary.renameTemplateTitle')}
        onClose={closeRename}
        onSubmit={submitRename}
      />
      <ConfirmDialog
        body={t('workflowLibrary.deleteConfirmBody', { name: deleteTarget?.name ?? '' })}
        confirmLabel={t('workflowLibrary.delete')}
        isOpen={deleteTarget !== null}
        title={t('workflowLibrary.deleteConfirmTitle')}
        onClose={closeDeleteConfirm}
        onConfirm={confirmDelete}
      />
    </>
  );

  return { actionItems, dialogs, open, openLabel, projectCopies };
};
