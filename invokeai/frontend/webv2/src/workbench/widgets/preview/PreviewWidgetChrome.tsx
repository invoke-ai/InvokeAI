import type { GalleryItem } from '@features/gallery';
import type { WidgetLabelProps, WorkbenchRegion } from '@workbench/widgetContracts';

import { Box, HStack, Text } from '@chakra-ui/react';
import {
  GalleryPickerPopover,
  type GalleryPickerAccept,
  type GalleryPickerOpeningScope,
} from '@features/gallery/picker';
import { Button, IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { isWidgetRegion } from '@workbench/layoutContracts';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { ChevronDownIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { usePreviewHeaderContext } from './previewHeaderStore';

/**
 * Show board/image context from previewHeaderStore, falling back to Preview except where the center trigger
 * already names it. The named item opens a mini gallery at that item's board to select another item, and carries a
 * control that clears the gallery selection it shows.
 */
export const PreviewWidgetLabel = ({ region }: WidgetLabelProps) => {
  const { t } = useTranslation();
  const { actionItem, boardName, itemName } = usePreviewHeaderContext();

  if (!actionItem || !itemName || !boardName) {
    if (region === 'center') {
      return null;
    }

    return (
      <Text fontSize="md" fontWeight="700">
        {t('widgets.labels.preview')}
      </Text>
    );
  }

  // Pad the truncated label away from the first header action in a panel header. In the center the label ends its
  // chrome island, whose own padding sets the clear button's inset like the island's other edge icon buttons.
  return (
    <HStack flex="1" gap="1" minW="0" pe={region === 'center' ? undefined : '2'}>
      <PreviewItemPicker boardName={boardName} item={actionItem} itemName={itemName} />
      <ClearSelectionButton region={region} />
    </HStack>
  );
};

const PREVIEW_PICKER_ACCEPT: GalleryPickerAccept = ['image', 'video'];

/**
 * The trigger shrinks before anything overlaps, keeping the item name longest: the board name truncates first, then
 * "Board /" drops out as a whole, and only then does the item name middle-truncate. The drop is a one-line wrapping row
 * clipped to its height, laid out in reverse: the name group comes first in the DOM, so it is the board group that
 * wraps onto the clipped second line once the two floors no longer fit side by side, while `row-reverse` still shows
 * "Board / name" from the left. Above the floors the name takes the free space first, up to its own width. CSS alone,
 * so no measuring on resize.
 */
const NAME_ROW_CSS = {
  alignContent: 'flex-start',
  alignItems: 'center',
  columnGap: '1',
  display: 'flex',
  flexDirection: 'row-reverse',
  flexWrap: 'wrap',
  // One line of the trigger's text; the wrapped board group sits below it, out of view.
  height: '1lh',
  // Main-end of a reversed row: packs the line against the left edge.
  justifyContent: 'flex-end',
  minW: '0',
  overflow: 'hidden',
} as const;
// The floors decide when the board drops: room for a few head characters plus the name's kept tail, and a few
// characters of board. The lopsided grow weights hand the name all free space until its max-content clamp freezes it;
// a modest ratio would leave it a fraction of a pixel short, and ellipsized, while the board still grows.
const ITEM_NAME_CSS = { flex: '1000000 1 16ch', maxW: 'max-content', minW: '0' } as const;
const BOARD_GROUP_CSS = { columnGap: '1', display: 'flex', flex: '1 1 8ch', maxW: 'max-content', minW: '0' } as const;

/**
 * The board and item name as the trigger of a mini gallery that opens at the item's own board and view, with the item
 * highlighted. The picker browses any board, so a pick is selected within its own board: Preview's next/previous then
 * step through that board rather than the listing the Gallery shows.
 */
const PreviewItemPicker = ({
  boardName,
  item,
  itemName,
}: {
  boardName: string;
  item: GalleryItem;
  itemName: string;
}) => {
  const { t } = useTranslation();
  const { gallery } = useWorkbenchCommands();
  const triggerIds = useTooltipTriggerIds();
  const rowRef = useRef<HTMLDivElement>(null);
  const [isTooltipOpen, setIsTooltipOpen] = useState(false);
  const selectItem = useCallback((picked: GalleryItem) => gallery.selectItemInItsBoard(picked), [gallery]);
  // Generated media lists under Images, uploads under Assets: open on the view that shows the item.
  const openingScope = useMemo<GalleryPickerOpeningScope>(
    () => ({ boardId: item.boardId, galleryView: item.category === 'general' ? 'images' : 'assets' }),
    [item.boardId, item.category]
  );
  // The tooltip names the item only while the row hides some of it; checked when it asks to open, so a resize costs
  // no render.
  const handleTooltipOpenChange = useCallback(
    ({ open }: { open: boolean }) => setIsTooltipOpen(open && !!rowRef.current && isRowClipped(rowRef.current)),
    []
  );
  const label = t('widgets.preview.boardAndItem', { board: boardName, name: itemName });

  return (
    <GalleryPickerPopover
      accept={PREVIEW_PICKER_ACCEPT}
      ids={triggerIds}
      label={t('widgets.gallery.picker.chooseMedia')}
      openingScope={openingScope}
      onPick={selectItem}
    >
      <Tooltip content={label} ids={triggerIds} open={isTooltipOpen} onOpenChange={handleTooltipOpenChange}>
        <Button
          aria-label={t('widgets.preview.chooseAnotherItem', { label })}
          // The button recipe refuses to shrink; the label gives way instead of pushing over the clear button.
          flexShrink={1}
          fontSize="md"
          gap="1"
          minW="0"
          overflow="hidden"
          px="1.5"
          size="sm"
          variant="ghost"
        >
          <Box ref={rowRef} css={NAME_ROW_CSS}>
            <MiddleTruncate
              css={ITEM_NAME_CSS}
              color="fg.muted"
              fontWeight="normal"
              text={itemName}
              title={undefined}
            />
            <Box css={BOARD_GROUP_CSS}>
              <Text flexShrink={1} fontWeight="700" minW="0" truncate>
                {boardName}
              </Text>
              <Text color="fg.muted" flexShrink={0} fontWeight="normal">
                /
              </Text>
            </Box>
          </Box>
          <ChevronDownIcon aria-hidden size={12} style={CHEVRON_STYLE} />
        </Button>
      </Tooltip>
    </GalleryPickerPopover>
  );
};

/** True when the board group has wrapped out of the row's one visible line, or any of its text is truncated. */
const isRowClipped = (row: HTMLElement): boolean => {
  const boardGroup = row.lastElementChild;
  if (!boardGroup || boardGroup.getBoundingClientRect().top >= row.getBoundingClientRect().bottom) {
    return true;
  }

  return [...row.querySelectorAll('*')].some((element) => element.scrollWidth > element.clientWidth);
};

const CHEVRON_STYLE = { flexShrink: 0 } as const;

const CLEAR_SELECTION_HOVER = { color: 'fg' } as const;

const ClearSelectionButton = ({ region }: { region: WorkbenchRegion }) => {
  const { t } = useTranslation();
  const { gallery } = useWorkbenchCommands();
  const { focusRegion } = useWorkbenchFocus();
  const label = t('widgets.gallery.commands.clearSelection');
  // The gallery's clear-selection command. This button leaves with the selection, so focus moves to the Preview's
  // region instead of falling to the document body.
  const clearSelection = useCallback(() => {
    if (isWidgetRegion(region)) {
      focusRegion(region, 'preview');
    }
    gallery.clearSelection();
  }, [focusRegion, gallery, region]);

  return (
    <IconButton
      _hover={CLEAR_SELECTION_HOVER}
      aria-label={label}
      color="fg.muted"
      flexShrink={0}
      size="sm"
      tooltip={label}
      variant="ghost"
      onClick={clearSelection}
    >
      <XIcon />
    </IconButton>
  );
};
