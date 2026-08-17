import type { ReactNode } from 'react';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { MenuContent } from '@platform/ui/Menu';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { Link } from '@tanstack/react-router';
import { ArrowRightIcon, CopyIcon, FileDownIcon, PencilIcon, PinIcon, PinOffIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ProjectCardActions } from './useProjectCardActions';

/**
 * The per-project action menu, plus the rename and delete dialogs it opens.
 * The grid card and the list row both mount this, so right-click and the
 * overflow button offer the same things in the same order everywhere.
 */

const MENU_ITEM_DELETE_HOVER = { bg: 'bg.error', color: 'fg.error' } as const;

export interface ProjectActionsMenuProps {
  actions: ProjectCardActions;
  isPinned: boolean;
  /** Anchor point for a right-click; `null` anchors to the trigger instead. */
  contextMenuTarget: { x: number; y: number } | null;
  isOpen: boolean;
  projectId: string;
  projectName: string;
  children: ReactNode;
  onOpenChange: (details: { open: boolean }) => void;
  onTogglePin: () => void;
}

const MENU_POSITION_BOTTOM_END = { placement: 'bottom-end' } as const;

export const ProjectActionsMenu = ({
  actions,
  children,
  contextMenuTarget,
  isOpen,
  isPinned,
  onOpenChange,
  onTogglePin,
  projectId,
  projectName,
}: ProjectActionsMenuProps) => {
  const { t } = useTranslation();
  const [isRenameOpen, setIsRenameOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const projectSearch = useMemo(() => ({ project: projectId }), [projectId]);
  const positioning = useMemo(
    () =>
      contextMenuTarget
        ? {
            getAnchorRect: () => ({ height: 1, width: 1, x: contextMenuTarget.x, y: contextMenuTarget.y }),
            placement: 'bottom-start' as const,
          }
        : MENU_POSITION_BOTTOM_END,
    [contextMenuTarget]
  );

  const openRenameDialog = useCallback(() => setIsRenameOpen(true), []);
  const closeRenameDialog = useCallback(() => setIsRenameOpen(false), []);
  const openDeleteDialog = useCallback(() => setIsDeleteOpen(true), []);
  const closeDeleteDialog = useCallback(() => setIsDeleteOpen(false), []);
  const handleDuplicate = useCallback(() => void actions.duplicate(), [actions]);
  const handleExport = useCallback(() => void actions.export(), [actions]);

  return (
    <>
      <Menu.Root open={isOpen} positioning={positioning} onOpenChange={onOpenChange}>
        {children}
        <Portal>
          <Menu.Positioner>
            <MenuBody
              isPinned={isPinned}
              projectSearch={projectSearch}
              onDelete={openDeleteDialog}
              onDuplicate={handleDuplicate}
              onExport={handleExport}
              onRename={openRenameDialog}
              onTogglePin={onTogglePin}
            />
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <RenameDialog
        initialName={projectName}
        isOpen={isRenameOpen}
        label={t('projects.renameProjectNameLabel')}
        submitLabel={t('common.rename')}
        title={t('projects.renameProject')}
        onClose={closeRenameDialog}
        onSubmit={actions.rename}
      />

      <ConfirmDialog
        // No "it is open, so it may come back" caveat any more: an open project is deleted through
        // the editor's own sync handle, so the deletion is final either way. What is worth saying
        // instead is what else goes — the project's board — and what does not.
        body={`${t('projects.deleteProjectCardBody', { name: projectName })} ${t('projects.deleteProjectBoardNote')}`}
        confirmLabel={t('projects.deleteProject')}
        isOpen={isDeleteOpen}
        title={t('projects.deleteProjectQuestion')}
        onClose={closeDeleteDialog}
        onConfirm={actions.delete}
      />
    </>
  );
};

const MenuBody = ({
  isPinned,
  onDelete,
  onDuplicate,
  onExport,
  onRename,
  onTogglePin,
  projectSearch,
}: {
  isPinned: boolean;
  projectSearch: { project: string };
  onDelete: () => void;
  onDuplicate: () => void;
  onExport: () => void;
  onRename: () => void;
  onTogglePin: () => void;
}) => {
  const { t } = useTranslation();

  return (
    <MenuContent minW="44">
      <Menu.Item asChild value="open">
        <Link search={projectSearch} to="/app">
          <Icon as={ArrowRightIcon} boxSize="3.5" />
          {t('common.open')}
        </Link>
      </Menu.Item>
      <Menu.Item value="pin" onClick={onTogglePin}>
        <Icon as={isPinned ? PinOffIcon : PinIcon} boxSize="3.5" />
        {isPinned ? t('projects.unpin') : t('projects.pin')}
      </Menu.Item>
      <Menu.Separator />
      <Menu.Item value="rename" onClick={onRename}>
        <Icon as={PencilIcon} boxSize="3.5" />
        {t('projects.renameWithEllipsis')}
      </Menu.Item>
      <Menu.Item value="duplicate" onClick={onDuplicate}>
        <Icon as={CopyIcon} boxSize="3.5" />
        {t('common.duplicate')}
      </Menu.Item>
      <Menu.Item value="export" onClick={onExport}>
        <Icon as={FileDownIcon} boxSize="3.5" />
        {t('common.export')}
      </Menu.Item>
      <Menu.Separator />
      <Menu.Item color="fg.error" value="delete" _hover={MENU_ITEM_DELETE_HOVER} onClick={onDelete}>
        <Icon as={Trash2Icon} boxSize="3.5" />
        {t('common.delete')}…
      </Menu.Item>
    </MenuContent>
  );
};
