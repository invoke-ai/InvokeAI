import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';

import { Menu, Portal } from '@chakra-ui/react';
import { MenuContent } from '@platform/ui/Menu';
import { memo, useCallback, useMemo, useState, type ReactNode } from 'react';

import { useWorkflowLibraryEntryActions, type WorkflowLibraryDuplicateGuard } from './useWorkflowLibraryEntryActions';
import {
  getWorkflowLibraryCardId,
  getWorkflowLibraryCardMenuContentId,
  keepCardMenuOpenForRetarget,
  keepMenuOpenWhenSiblingMenuCloses,
  type WorkflowCardMenuAnchor,
} from './WorkflowLibraryCard';

/**
 * The workflow a card's or row's actions menu was raised for, and the control that raised it. It outlives the open
 * menu: the menu learns it closed only after React has rendered the close, so the trigger it restores focus to must
 * still be the one that opened it, and the dialogs its items open belong to the same workflow.
 */
export interface WorkflowContextMenuTarget {
  /** The tile's or row's More button when it raised the menu; null for a right-click or Shift+F10 on the card. */
  triggerId: string | null;
  workflowId: string;
}

export interface WorkflowCardContextMenuProps {
  /** Where the open menu is anchored; null while it is closed. */
  anchor: WorkflowCardMenuAnchor | null;
  children: ReactNode;
  target: WorkflowContextMenuTarget | null;
  onClose: () => void;
}

/**
 * A card's or row's actions menu, for the workflow it was raised on whether or not that is the selection. Naming the
 * card (or its More button) as the trigger makes the menu a nested layer of the library dialog: the dialog's focus
 * trap then lets the menu keep focus, and closing returns it there. A pointer point is a fixed rect; a More button
 * anchors the menu itself, so the menu follows it as the collection scrolls.
 */
export const WorkflowCardContextMenu = ({ anchor, children, target, onClose }: WorkflowCardContextMenuProps) => {
  /*
   * A press on another control (the rail's More, the search field) closes the menu and keeps focus there. Zag skips
   * focus return for such a press only on an uncontrolled close; this menu is controlled, so it names no trigger to
   * return to until it opens again.
   */
  const [returnsFocus, setReturnsFocus] = useState(true);
  const [openedAt, setOpenedAt] = useState(anchor);
  if (anchor !== openedAt) {
    setOpenedAt(anchor);
    if (anchor !== null) {
      setReturnsFocus(true);
    }
  }
  const handlePointerDownOutside = useCallback(
    (event: Parameters<NonNullable<Menu.RootProps['onPointerDownOutside']>>[0]) => {
      keepCardMenuOpenForRetarget(event);
      if (!event.defaultPrevented && event.detail.focusable) {
        setReturnsFocus(false);
      }
    },
    []
  );
  const ids = useMemo(() => {
    if (!target) {
      return undefined;
    }
    const content = getWorkflowLibraryCardMenuContentId(target.workflowId);

    return {
      content,
      trigger: returnsFocus ? (target.triggerId ?? getWorkflowLibraryCardId(target.workflowId)) : `${content}-closed`,
    };
  }, [returnsFocus, target]);
  const positioning = useMemo(() => {
    // The menu belongs to the library dialog's visible bounds, rather than the whole viewport.
    const boundary = () =>
      (ids ? document.getElementById(ids.trigger) : null)?.closest<HTMLElement>(
        '[data-scope="dialog"][data-part="content"]'
      ) ?? [];

    return anchor?.kind === 'point'
      ? {
          boundary,
          getAnchorRect: () => ({ height: 1, width: 1, x: anchor.x, y: anchor.y }),
          placement: 'bottom-start' as const,
        }
      : { boundary, placement: 'bottom-end' as const };
  }, [anchor, ids]);
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Menu.Root
      ids={ids}
      open={anchor !== null}
      positioning={positioning}
      onOpenChange={handleOpenChange}
      onPointerDownOutside={handlePointerDownOutside}
      onRequestDismiss={keepMenuOpenWhenSiblingMenuCloses}
    >
      <Portal>
        <Menu.Positioner>
          <MenuContent
            data-workflow-context-menu
            minW="min(13rem, var(--available-width))"
            maxW="var(--available-width)"
          >
            {children}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

export interface WorkflowLibraryContextMenuProps {
  anchor: WorkflowCardMenuAnchor | null;
  duplicateGuard: WorkflowLibraryDuplicateGuard;
  /** The target's entry; null once it has left the resident window, when the shell has already closed the menu. */
  entry: WorkflowLibraryEntry | null;
  isReadOnly: boolean;
  projectWorkflows: readonly ProjectWorkflowEntry[];
  target: WorkflowContextMenuTarget | null;
  /** The shell closes the library when a fork takes the user to a new project. */
  onClose: () => void;
  onDeleted: (workflowId: string) => void;
  onDuplicated: (workflowId: string) => void;
  onMenuClose: () => void;
  onOpen: (item: WorkflowLibraryListItem) => void;
}

/** A library card's actions menu with the dialogs its items open; memoised like the rail, which browsing re-renders. */
export const WorkflowLibraryContextMenu = memo(function WorkflowLibraryContextMenu({
  anchor,
  duplicateGuard,
  entry,
  isReadOnly,
  projectWorkflows,
  target,
  onClose,
  onDeleted,
  onDuplicated,
  onMenuClose,
  onOpen,
}: WorkflowLibraryContextMenuProps) {
  const { actionItems, dialogs } = useWorkflowLibraryEntryActions({
    duplicateGuard,
    entry,
    isReadOnly,
    projectWorkflows,
    onClose,
    onDeleted,
    onDuplicated,
    onOpen,
  });

  return (
    <>
      <WorkflowCardContextMenu anchor={anchor} target={target} onClose={onMenuClose}>
        {actionItems}
      </WorkflowCardContextMenu>
      {dialogs}
    </>
  );
});
