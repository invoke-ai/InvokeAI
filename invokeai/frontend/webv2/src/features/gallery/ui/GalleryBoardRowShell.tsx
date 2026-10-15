import type { DraggableSyntheticListeners } from '@dnd-kit/core';

import { HStack, Stack, type SystemStyleObject } from '@chakra-ui/react';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Row } from '@platform/ui/Row';
import { useMemo, type MouseEvent, type ReactNode, type Ref } from 'react';

/** On the container, not the row button: the `⋮` sibling must share the fill. */
const ROW_CONTAINER_CSS = {
  '&:hover .board-row-actions, &:focus-within .board-row-actions': { opacity: 1 },
  '&:hover .board-row-hover, &:focus-within .board-row-hover': { opacity: 1 },
  // A translucent tint, not an opaque surface: the picker popover's own background is bg.muted, which hid one there.
  _hover: { bg: 'bg.hover' },
  borderRadius: 'sm',
  transition: 'background var(--wb-motion-duration-fast) ease',
} as const;

/** The active row is filled, matching `rowRecipe`'s own `active="accent"` variant. */
const SELECTED_CONTAINER_CSS: SystemStyleObject = {
  ...ROW_CONTAINER_CSS,
  _hover: { bg: 'accent.solid' },
  bg: 'accent.solid',
  color: 'accent.contrast',
} as const;

/** A row that can be dragged: touch scrolls the list until a hold arms the drag, which the row shows by losing colour. */
const DRAGGABLE_CSS: SystemStyleObject = { '&[data-drag-armed=true]': { filter: 'saturate(0)' }, touchAction: 'pan-y' };

/** Suppress hover fill during compatible drags so it cannot obscure the row's drop tint. */
const DROP_TARGET_CONTAINER_CSS: SystemStyleObject = {
  ...ROW_CONTAINER_CSS,
  _hover: { bg: 'transparent' },
} as const;

/** The container owns the fill, so the button contributes only its focus ring. */
const ROW_BUTTON_CSS = { _hover: { bg: 'transparent' }, bg: 'transparent' } as const;

/** Keep action buttons beside the row button to avoid invalid nesting and fragmented touch targets. */
export const GalleryBoardRowShell = ({
  actions,
  ariaLabel,
  children,
  cover,
  dragListeners,
  isDropTarget = false,
  isSelected = false,
  label,
  labelWeight,
  subtitle,
  ref,
  onContextMenu,
  onSelect,
}: {
  /** Hover-revealed trailing controls, outside the row button. */
  actions?: ReactNode;
  ariaLabel?: string;
  /** Trailing metadata rendered inside the row button (dates, counts, badges). */
  children?: ReactNode;
  cover: ReactNode;
  /** Lets the whole row, not only its button, be picked up and dragged. */
  dragListeners?: DraggableSyntheticListeners;
  /** A compatible drag is in flight and this row could receive it. */
  isDropTarget?: boolean;
  isSelected?: boolean;
  label: string;
  labelWeight?: string;
  subtitle?: ReactNode;
  ref?: Ref<HTMLDivElement>;
  onContextMenu?: (event: MouseEvent) => void;
  onSelect: () => void;
}) => {
  const isDraggable = dragListeners !== undefined;
  const containerCss = useMemo(() => {
    const css = isSelected ? SELECTED_CONTAINER_CSS : isDropTarget ? DROP_TARGET_CONTAINER_CSS : ROW_CONTAINER_CSS;

    return isDraggable ? { ...css, ...DRAGGABLE_CSS } : css;
  }, [isDraggable, isDropTarget, isSelected]);

  return (
    <HStack ref={ref} css={containerCss} gap="0" pe="1" w="full" {...dragListeners}>
      <Row active="none" asChild css={ROW_BUTTON_CSS} flex="1" gap="2" minH="7" minW="0" px="1" py="1">
        <button
          aria-current={isSelected ? 'true' : undefined}
          aria-label={ariaLabel}
          type="button"
          onClick={onSelect}
          onContextMenu={onContextMenu}
        >
          {cover}
          <Stack align="stretch" flex="1" gap="0" minW="0" textAlign="start">
            <MiddleTruncate fontWeight={labelWeight ?? (isSelected ? '600' : '500')} minW="0" text={label} />
            {subtitle}
          </Stack>
          {children}
        </button>
      </Row>
      {actions}
    </HStack>
  );
};
