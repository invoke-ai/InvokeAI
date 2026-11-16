/* oxlint-disable react-perf/jsx-no-jsx-as-prop -- ListItem slots take JSX; the React Compiler memoizes them. */
import type { Announcements, DragEndEvent, KeyboardSensorOptions } from '@dnd-kit/core';
import type { ReactNode } from 'react';

import { chakra, Icon } from '@chakra-ui/react';
import { closestCenter, DndContext, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { restrictToParentElement, restrictToVerticalAxis } from '@dnd-kit/modifiers';
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useMountEffect } from '@platform/react/useMountEffect';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { GripVerticalIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ListItem, type ListItemProps } from './ListItem';
import { ListStack, type ListStackProps } from './ListStack';

/*
 * Reordering lives apart from ListStack and ListItem, which load with the editor: only lazily loaded lists import this
 * module, so its code stays out of the startup chunks.
 */

const MODIFIERS = [restrictToVerticalAxis, restrictToParentElement];
const POINTER_SENSOR_OPTIONS = { activationConstraint: { distance: 6 } } as const;
const KEYBOARD_SENSOR_OPTIONS = { coordinateGetter: sortableKeyboardCoordinates } satisfies KeyboardSensorOptions;

export interface SortableListStackProps<Id extends string> extends Omit<ListStackProps, 'children'> {
  /** Row ids in their current order; each `SortableListItem` child takes one. */
  items: readonly Id[];
  /** One completed move: the dragged row takes the place of the row it was dropped on. */
  onReorder: (activeId: Id, overId: Id) => void;
  /** `SortableListItem` rows, in `items` order. */
  children: ReactNode;
}

interface RowData {
  label?: string;
  sortable?: { index: number; items: unknown[] };
}

const rowPosition = (entity: { data: { current?: unknown } } | null) => {
  const data = entity?.data.current as RowData | undefined;

  return data?.sortable
    ? { count: data.sortable.items.length, name: data.label ?? '', position: data.sortable.index + 1 }
    : null;
};

/**
 * A short list whose rows the user reorders by a handle, with the pointer or the keyboard (Space or Enter picks a row
 * up, the arrows move it, Space or Enter drops it, Escape cancels). Moves are announced by row title and position.
 */
export const SortableListStack = <Id extends string>({
  children,
  items,
  onReorder,
  ...stackProps
}: SortableListStackProps<Id>) => {
  const { t } = useTranslation();
  const [isSorting, setIsSorting] = useState(false);
  const sensors = useSensors(
    useSensor(PointerSensor, POINTER_SENSOR_OPTIONS),
    useSensor(KeyboardSensor, KEYBOARD_SENSOR_OPTIONS)
  );
  // dnd-kit reports the row over its own place as the drag starts; only a change of place is news.
  const announcedPosition = useRef(0);
  const accessibility = useMemo(() => {
    const announcements: Announcements = {
      onDragCancel: ({ active }) => {
        const row = rowPosition(active);
        return row ? t('common.list.reorderCancelled', row) : undefined;
      },
      onDragEnd: ({ active, over }) => {
        const row = rowPosition(active);
        const target = rowPosition(over);
        if (!row) {
          return undefined;
        }
        return target
          ? t('common.list.reorderDropped', { ...row, position: target.position })
          : t('common.list.reorderCancelled', row);
      },
      onDragOver: ({ active, over }) => {
        const row = rowPosition(active);
        const target = rowPosition(over);
        if (!row || !target || target.position === announcedPosition.current) {
          return undefined;
        }
        announcedPosition.current = target.position;
        return t('common.list.reorderMoved', { ...row, position: target.position });
      },
      onDragStart: ({ active }) => {
        const row = rowPosition(active);
        announcedPosition.current = row?.position ?? 0;
        return row ? t('common.list.reorderPickedUp', row) : undefined;
      },
    };

    return { announcements, screenReaderInstructions: { draggable: t('common.list.reorderInstructions') } };
  }, [t]);
  const sortableItems = useMemo(() => [...items], [items]);

  const isSortingRef = useRef(false);
  // Escape cancels the sort, and only the sort: claimed ahead of a dialog or popover around the list, whose own
  // capturing Escape listener would otherwise close it. dnd-kit's sensors still see the key.
  useMountEffect(() => {
    const claimEscape = (event: KeyboardEvent) => {
      if (isSortingRef.current && event.key === 'Escape') {
        event.preventDefault();
      }
    };
    window.addEventListener('keydown', claimEscape, true);
    return () => window.removeEventListener('keydown', claimEscape, true);
  });

  const setSorting = useCallback((sorting: boolean) => {
    isSortingRef.current = sorting;
    setIsSorting(sorting);
  }, []);
  const handleDragStart = useCallback(() => setSorting(true), [setSorting]);
  const handleDragCancel = useCallback(() => setSorting(false), [setSorting]);
  const handleDragEnd = useCallback(
    ({ active, over }: DragEndEvent) => {
      setSorting(false);
      if (over && active.id !== over.id) {
        onReorder(active.id as Id, over.id as Id);
      }
    },
    [onReorder, setSorting]
  );

  return (
    <DndContext
      accessibility={accessibility}
      collisionDetection={closestCenter}
      modifiers={MODIFIERS}
      sensors={sensors}
      onDragCancel={handleDragCancel}
      onDragEnd={handleDragEnd}
      onDragStart={handleDragStart}
    >
      <SortableContext items={sortableItems} strategy={verticalListSortingStrategy}>
        <ListStack data-sorting={isSorting || undefined} {...stackProps}>
          {children}
        </ListStack>
      </SortableContext>
    </DndContext>
  );
};

/**
 * The handle sits in the row's leading slot, so the row's primary area must stay a plain container: the props that
 * make it a button (press, menu, List focus key) are left out, and the row acts through its `actions`.
 */
export interface SortableListItemProps extends Omit<
  ListItemProps,
  | 'isExpanded'
  | 'isMenuOpen'
  | 'itemKey'
  | 'onContextMenu'
  | 'onIntent'
  | 'onPress'
  | 'pressLabel'
  | 'role'
  | 'tabIndex'
> {
  id: string;
  /** Accessible name of the drag handle; name the row ("Reorder {title}"). */
  handleLabel: string;
}

const SHADOW_TRANSITION = 'box-shadow var(--wb-motion-duration-fast) ease';

const ROW_CSS = {
  borderRadius: 'sm',
  position: 'relative',
  transition: SHADOW_TRANSITION,
  // Rows slide past the hairlines while sorting, so the hairlines step aside until the drop.
  '[data-sorting] > & + [data-list-divider]': { opacity: 0 },
  // The picked-up row lifts above the rows it passes, by pointer or keyboard alike.
  '&[data-dragging]': { bg: 'bg.panel', boxShadow: 'lg', zIndex: 1 },
} as const;

const HANDLE_CSS = {
  color: 'fg.subtle',
  cursor: 'grab',
  flexShrink: 0,
  // Sits nearer the row's edge than its content does.
  ms: '-1',
  touchAction: 'none',
};
const DRAGGING_HANDLE_CSS = { ...HANDLE_CSS, cursor: 'grabbing' };

/** A `ListItem` with a drag handle at its start, for a `SortableListStack`. */
export const SortableListItem = ({
  handleLabel,
  id,
  isDisabled = false,
  leading,
  ...rowProps
}: SortableListItemProps) => {
  const { t } = useTranslation();
  const { attributes, isDragging, listeners, setActivatorNodeRef, setNodeRef, transform, transition } = useSortable({
    attributes: { roleDescription: t('common.list.reorderRole') },
    data: { label: rowProps.title },
    disabled: isDisabled,
    id,
  });
  const style = useMemo(
    () => ({
      transform: CSS.Translate.toString(transform),
      // dnd-kit's own timing ignores the app's Reduce motion setting; the motion token follows it.
      transition: transition ? `transform var(--wb-motion-duration-slow) ease, ${SHADOW_TRANSITION}` : undefined,
    }),
    [transform, transition]
  );

  return (
    <chakra.div
      ref={setNodeRef}
      css={ROW_CSS}
      data-dragging={isDragging || undefined}
      data-list-row={id}
      role="listitem"
      style={style}
    >
      <ListItem
        {...rowProps}
        isDisabled={isDisabled}
        leading={
          <>
            {/* Quiet while its row moves: the tip would trail the drag. */}
            <Tooltip content={handleLabel} disabled={isDragging} keepsRootWhenDisabled>
              <IconButton
                ref={setActivatorNodeRef}
                {...attributes}
                {...listeners}
                aria-label={handleLabel}
                css={isDragging ? DRAGGING_HANDLE_CSS : HANDLE_CSS}
                disabled={isDisabled}
                size="sm"
                variant="ghost"
              >
                <Icon as={GripVerticalIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
            {leading}
          </>
        }
        role="presentation"
      />
    </chakra.div>
  );
};
