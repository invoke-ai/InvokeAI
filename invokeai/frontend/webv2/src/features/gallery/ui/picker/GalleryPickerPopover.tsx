import type { GalleryItem } from '@features/gallery/core/items';
import type { ReactElement } from 'react';

import { Popover, Portal, Skeleton, Stack } from '@chakra-ui/react';
import { PopoverContent } from '@platform/ui/Popover';
import { lazy, Suspense, useCallback, useRef, useState } from 'react';

import type { GalleryPickerAccept, GalleryPickerSelection } from './galleryPicker';
import type { GalleryPickerOpeningScope } from './useGalleryPickerScope';

// Loaded on first open: every widget with an image slot mounts a trigger at
// boot, and the view is most of the picker's weight.
const GalleryPickerView = lazy(() =>
  import('./GalleryPickerView').then((module) => ({ default: module.GalleryPickerView }))
);

const POSITIONING = { placement: 'bottom-start' } as const;

/** Same header and search footprint as the view, so the popover does not jump when it lands. */
const PICKER_FALLBACK = (
  <Stack aria-busy="true" gap="2" p="2">
    <Skeleton h="control.md" rounded="control" />
    <Skeleton h="control.md" rounded="control" />
    <Skeleton h="10rem" rounded="md" />
  </Stack>
);
const SINGLE_SELECTION: GalleryPickerSelection = { mode: 'single' };

export interface GalleryPickerPopoverProps {
  accept: GalleryPickerAccept;
  /** The trigger element; must accept a ref and a click handler. */
  children: ReactElement;
  /** Shared with a Tooltip wrapping an icon-only trigger (see `useTooltipTriggerIds`). */
  ids?: { trigger: string };
  /** The dialog's accessible name, e.g. "Choose image". */
  label: string;
  /**
   * The board and view each opening starts at, read once per opening: a host showing one item opens where that item
   * lives. Absent, the picker opens at the Gallery widget's board and view.
   */
  openingScope?: GalleryPickerOpeningScope;
  selection?: GalleryPickerSelection;
  onPick: (item: GalleryItem) => void;
}

/**
 * Mount on open and unmount on close so each opening starts fresh from `openingScope`, or the Gallery widget's scope
 * without one; browsing inside the picker never carries over to the next opening.
 */
export const GalleryPickerPopover = ({
  accept,
  children,
  ids,
  label,
  openingScope,
  selection = SINGLE_SELECTION,
  onPick,
}: GalleryPickerPopoverProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const handleOpenChange = useCallback(({ open }: { open: boolean }) => setIsOpen(open), []);
  const close = useCallback(() => setIsOpen(false), []);
  // Null until the lazy view has mounted; the view then focuses the field itself.
  const getInitialFocusEl = useCallback(() => searchInputRef.current, []);

  return (
    <Popover.Root
      ids={ids}
      initialFocusEl={getInitialFocusEl}
      lazyMount
      open={isOpen}
      positioning={POSITIONING}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Popover.Trigger asChild>{children}</Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <PopoverContent
            aria-label={label}
            display="flex"
            flexDirection="column"
            maxH="min(26rem, var(--available-height))"
            p="0"
            w="clamp(20rem, var(--reference-width), 28rem)"
          >
            <Suspense fallback={PICKER_FALLBACK}>
              <GalleryPickerView
                accept={accept}
                openingScope={openingScope}
                searchInputRef={searchInputRef}
                selection={selection}
                onClose={close}
                onPick={onPick}
              />
            </Suspense>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
