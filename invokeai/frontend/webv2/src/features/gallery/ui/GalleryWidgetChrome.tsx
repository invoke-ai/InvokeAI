import type { GalleryBoard } from '@features/gallery/core/types';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { getGalleryBoardDestinationGroups, getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { getGallerySettings, type GallerySettings } from '@features/gallery/core/settings';
import { galleryBoardsOptions } from '@features/gallery/data/queries';
import { Button } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ScrollableMenuContent } from '@platform/ui/ScrollableMenuContent';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useQuery } from '@tanstack/react-query';
import {
  ChevronDownIcon,
  ChevronsDownUpIcon,
  ChevronsUpDownIcon,
  ImageIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  SquareArrowDownIcon,
} from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryWidgetProps } from './GalleryUiContext';

import { BoardCover, BoardCoverIcon } from './GalleryBoardCover';
import { getGalleryProjectBoardId, getGallerySelectedBoardId } from './galleryStateView';
import { useGalleryUi, useHostGalleryBoards } from './GalleryUiContext';
import { useGalleryDestination } from './useGalleryDestination';

type GalleryChromeProps = { region: GalleryWidgetProps['region'] };

const LABEL_EXPANDED_PROPS = { bg: 'transparent' } as const;
// Clipped, but with room for the board button's focus ring: the last resort on a header too narrow for its parts.
const LABEL_CLIP_CSS = { overflow: 'clip', overflowClipMargin: '4px' } as const;
/**
 * Measured against the header row, the nearest inline-size container (the widget header, or the center chrome): about
 * half of it, less 3rem because the label's fixed parts ("Gallery /", the board's cover and chevron) are wider than
 * Saving to's. The board and destination names then get about equal room and truncate rather than overlap.
 */
const DESTINATION_MAX_WIDTH = 'min(18rem, max(6.5rem, 50cqi - 3rem))';
/** On a narrow header the trigger shows only the board; its accessible name still starts with "Saving to". */
const DESTINATION_PREFIX_CSS = { '@container (width < 30rem)': { display: 'none' } } as const;
/**
 * Rails shrink with the viewport, past the size a drag can set. Measured on the header's content box (its border box
 * less 20px of padding): below 20rem the label drops "Gallery /" so the board keeps room to name itself; below 15rem
 * (a 920px window's rail) the destination also drops its name, keeping its icon and the chevron that marks it as a
 * menu rather than a download button. Both accessible names stay whole.
 */
const NARROW_HEADER_HIDDEN_CSS = { '@container (width < 20rem)': { display: 'none' } } as const;
const COMPACT_HEADER = '@container (width < 15rem)';
const COMPACT_HIDDEN_CSS = { [COMPACT_HEADER]: { display: 'none' } } as const;
/** Icon and chevron, inset like the header's icon buttons. */
const DESTINATION_COMPACT_CSS = { [COMPACT_HEADER]: { px: '1' } } as const;
/** The boards sidebar sits at the inline start, so its panel icon flips with the text direction. */
const MIRRORED_IN_RTL = { transform: 'scaleX(-1)' } as const;

const useGalleryChromeBoards = () => {
  const { gallery, galleryValues } = useGalleryUi();
  const settings = getGallerySettings(galleryValues);
  const { data: boards } = useQuery(
    galleryBoardsOptions({
      includeArchived: settings.showArchivedBoards,
      includeDateBoards: settings.showDateBoards,
      orderBy: settings.boardOrderBy,
      orderDir: settings.boardOrderDir,
    })
  );
  const resolvedBoards = useHostGalleryBoards(boards);

  return {
    boards: resolvedBoards,
    gallery,
    galleryValues,
    selectedBoardId: getGallerySelectedBoardId(galleryValues, resolvedBoards),
    settings,
  };
};

export const GalleryWidgetLabel = ({ region }: GalleryChromeProps) => {
  const { t } = useTranslation();
  const { projectId } = useGalleryUi();
  const { boards, gallery, selectedBoardId, settings } = useGalleryChromeBoards();
  const selectedBoard = boards.find((board) => board.id === selectedBoardId);
  // Another project's inbox is just "Inbox" in its own list; the header has no section to say whose.
  const boardName = !selectedBoard
    ? t('widgets.gallery.selectedBoardFallback')
    : selectedBoard.isInbox && selectedBoard.projectId !== projectId
      ? t('widgets.gallery.inboxOf', { project: selectedBoard.name })
      : getGalleryBoardLabel(selectedBoard, t);
  const isCollapsed = settings.boardPanelCollapsed;

  const toggleBoards = useCallback(
    () => gallery.updateSettings({ boardPanelCollapsed: !isCollapsed }),
    [gallery, isCollapsed]
  );

  return (
    <HStack css={LABEL_CLIP_CSS} flex="1" gap="1" minW="0" pe={region === 'center' ? undefined : '2'}>
      {region === 'center' ? null : (
        <Text css={NARROW_HEADER_HIDDEN_CSS} flexShrink={0} fontWeight="700" pe="2">
          {t('widgets.labels.gallery')}
        </Text>
      )}
      <Text color="fg.subtle" css={region === 'center' ? undefined : NARROW_HEADER_HIDDEN_CSS} flexShrink={0}>
        /
      </Text>
      {/* No tooltip: it renders over the button and steals its hover. */}
      <Button
        _expanded={LABEL_EXPANDED_PROPS}
        aria-expanded={!isCollapsed}
        aria-label={t('widgets.gallery.toggleBoardsNamed', { name: boardName })}
        // Buttons keep their width by default; this one gives way so the board's name truncates.
        flexShrink={1}
        maxW="14rem"
        minW="0"
        // The cover keeps the same 2px inset on every side of the 24px button.
        ps="0.5"
        size="sm"
        variant="ghost"
        onClick={toggleBoards}
      >
        {selectedBoard ? <BoardCover board={selectedBoard} /> : <BoardCoverIcon icon={ImageIcon} />}
        <MiddleTruncate fontWeight="600" minW="0" text={boardName} />
        {/* The center view shows the boards as a sidebar beside the grid; the rails stack them above it. */}
        {region === 'center' ? (
          <Icon
            _rtl={MIRRORED_IN_RTL}
            as={isCollapsed ? PanelLeftOpenIcon : PanelLeftCloseIcon}
            boxSize="3.5"
            color="fg.subtle"
            flexShrink={0}
          />
        ) : (
          <Icon
            as={isCollapsed ? ChevronsUpDownIcon : ChevronsDownUpIcon}
            boxSize="3"
            color="fg.subtle"
            flexShrink={0}
          />
        )}
      </Button>
    </HStack>
  );
};

const DESTINATION_MENU_POSITIONING = { placement: 'bottom-end' } as const;
const DESTINATION_UNCATEGORIZED = 'none';
/** The menu value of the default destination, which stores no choice: the project's inbox, else Uncategorized. */
const DESTINATION_DEFAULT = 'default';
/** Saving to lists no other project's boards, so it names none. */
const NO_PROJECT_NAMES: ReadonlyMap<string, string> = new Map();

/**
 * Where new generations, canvas saves and pasted uploads go, named in the header and changed from its menu: a fixed
 * board (the project's inbox unless one is chosen), or the selected board while "Follow the selected board" is on.
 * Choosing a board stops following; choosing the default stores no choice, so it follows the project's inbox.
 */
export const GalleryWidgetHeaderActions = () => {
  const { t } = useTranslation();
  const { gallery } = useGalleryUi();
  const { boards, galleryValues, selectedBoardId, settings } = useGalleryChromeBoards();
  const triggerIds = useTooltipTriggerIds();
  const projectBoardId = getGalleryProjectBoardId(galleryValues);
  const destination = useGalleryDestination({ boards, projectBoardId, selectedBoardId, settings });
  const getBoardName = (boardId: string) => {
    const board = destination.getBoard(boardId);

    return boardId === DESTINATION_UNCATEGORIZED
      ? t('widgets.gallery.uncategorized')
      : board
        ? getGalleryBoardLabel(board, t)
        : boardId === projectBoardId
          ? t('widgets.gallery.inbox')
          : t('widgets.gallery.unknownBoard');
  };
  const destinationName = getBoardName(destination.boardId);
  const tooltip =
    destination.refusedBoardId !== null
      ? t('widgets.gallery.destinationRedirectedTooltip', {
          destination: destinationName,
          name: getBoardName(destination.refusedBoardId),
        })
      : settings.followSelectedBoard
        ? t('widgets.gallery.destinationFollowingTooltip', { name: destinationName })
        : t('widgets.gallery.destinationTooltip', { name: destinationName });
  const toggleFollow = useCallback(
    (checked: boolean) => gallery.updateSettings({ followSelectedBoard: checked }),
    [gallery]
  );
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const handleMenuOpenChange = useCallback((details: { open: boolean }) => setIsMenuOpen(details.open), []);

  return (
    // Mounted only while open: its covers and groups cost nothing on the gallery writes the header re-renders for.
    <Menu.Root
      ids={triggerIds}
      lazyMount
      positioning={DESTINATION_MENU_POSITIONING}
      unmountOnExit
      onOpenChange={handleMenuOpenChange}
    >
      {/* Not over the menu it describes: it opens below the trigger, where the tooltip would cover the first item. */}
      <Tooltip content={tooltip} disabled={isMenuOpen} ids={triggerIds}>
        <Menu.Trigger asChild>
          <Button
            // Always starts with "Saving to" and the board, then the hint. The visible text ("Saving to" and the
            // board, or the board alone on a narrow header) is contained in it, so label-in-name holds in every
            // language and a speech user can say what the button shows.
            aria-label={`${t('widgets.gallery.savingTo')} ${destinationName}. ${t('widgets.gallery.destinationTriggerHint')}`}
            css={DESTINATION_COMPACT_CSS}
            data-gallery-destination={destination.boardId}
            maxW={DESTINATION_MAX_WIDTH}
            minW="0"
            size="sm"
            variant="ghost"
          >
            <Icon as={SquareArrowDownIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
            {/* One run of text, label and board joined by a space, so the visible text reads as the start of the
                accessible name ("Saving to Inbox"); separate flex items would read as "Saving toInbox". */}
            <HStack as="span" css={COMPACT_HIDDEN_CSS} gap="0" minW="0">
              <Text as="span" color="fg.muted" css={DESTINATION_PREFIX_CSS} flexShrink={0} whiteSpace="pre">
                {`${t('widgets.gallery.savingTo')} `}
              </Text>
              <MiddleTruncate as="span" fontWeight="600" minW="0" text={destinationName} />
            </HStack>
            <Icon as={ChevronDownIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
          </Button>
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <ScrollableMenuContent maxW="18rem" minW="14rem">
            <Menu.CheckboxItem
              checked={settings.followSelectedBoard}
              value="follow-selected-board"
              onCheckedChange={toggleFollow}
            >
              <Menu.ItemText>{t('widgets.gallery.followSelectedBoard')}</Menu.ItemText>
              <Menu.ItemIndicator />
            </Menu.CheckboxItem>
            <Menu.Separator />
            <DestinationChoices boards={boards} projectBoardId={projectBoardId} settings={settings} />
          </ScrollableMenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/**
 * The boards Saving to offers: the open project's and the Library's, with Uncategorized; another project's boards or a
 * shared board are chosen from their own menu. The default is stored as no choice.
 */
const DestinationChoices = ({
  boards,
  projectBoardId,
  settings,
}: {
  boards: readonly GalleryBoard[];
  projectBoardId: string | null;
  settings: GallerySettings;
}) => {
  const { t } = useTranslation();
  const { gallery, projectId, projectName } = useGalleryUi();
  const defaultId = projectBoardId ?? DESTINATION_UNCATEGORIZED;
  const toMenuValue = (boardId: string) => (boardId === defaultId ? DESTINATION_DEFAULT : boardId);
  const groups = getGalleryBoardDestinationGroups({
    boards: boards.filter((board) => board.kind === 'board' && !board.archived),
    projectId,
    projectName,
    projectNames: NO_PROJECT_NAMES,
    t,
  }).filter((group) => group.id === 'project' || group.id === 'library');
  // While following, no fixed destination is checked: the follow item is.
  const checkedValue = settings.followSelectedBoard
    ? ''
    : settings.autoAddBoardId === null
      ? DESTINATION_DEFAULT
      : toMenuValue(settings.autoAddBoardId);
  const chooseDestination = useCallback(
    (details: { value: string }) =>
      gallery.updateSettings({
        autoAddBoardId: details.value === DESTINATION_DEFAULT ? null : details.value,
        followSelectedBoard: false,
      }),
    [gallery]
  );
  const uncategorized = (
    <Menu.RadioItem value={toMenuValue(DESTINATION_UNCATEGORIZED)}>
      <BoardCoverIcon icon={ImageIcon} />
      <Menu.ItemText>{t('widgets.gallery.uncategorized')}</Menu.ItemText>
      <Menu.ItemIndicator />
    </Menu.RadioItem>
  );

  return (
    <>
      {groups.map((group) => (
        <Menu.ItemGroup key={group.id}>
          <Menu.ItemGroupLabel>{group.label}</Menu.ItemGroupLabel>
          <Menu.RadioItemGroup value={checkedValue} onValueChange={chooseDestination}>
            {group.boards.map((board) => (
              <DestinationItem key={board.id} board={board} value={toMenuValue(board.id)} />
            ))}
            {group.id === 'library' ? uncategorized : null}
          </Menu.RadioItemGroup>
        </Menu.ItemGroup>
      ))}
      {groups.some((group) => group.id === 'library') ? null : (
        <Menu.ItemGroup>
          <Menu.ItemGroupLabel>{t('widgets.gallery.boardGroups.library')}</Menu.ItemGroupLabel>
          <Menu.RadioItemGroup value={checkedValue} onValueChange={chooseDestination}>
            {uncategorized}
          </Menu.RadioItemGroup>
        </Menu.ItemGroup>
      )}
    </>
  );
};

const DestinationItem = ({ board, value }: { board: GalleryBoard; value: string }) => {
  const { t } = useTranslation();
  const label = getGalleryBoardLabel(board, t);
  const isDefault = value === DESTINATION_DEFAULT;

  return (
    <Menu.RadioItem value={value}>
      <BoardCover board={board} />
      <MiddleTruncate flex="1" fontSize="md" minW="0" text={label} />
      {isDefault ? (
        <Text color="fg.subtle" flexShrink={0} fontSize="sm">
          {t('widgets.gallery.destinationDefault')}
        </Text>
      ) : null}
      <Menu.ItemIndicator />
    </Menu.RadioItem>
  );
};
