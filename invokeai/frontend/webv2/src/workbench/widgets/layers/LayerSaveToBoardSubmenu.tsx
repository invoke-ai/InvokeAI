import type { GalleryBoard } from '@features/gallery';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { getGalleryBoardLabel, getGallerySettings } from '@features/gallery/contracts';
import { galleryBoardsOptions, getGalleryListingBoardsQuery } from '@features/gallery/queries';
import { useHostGalleryBoards } from '@features/gallery/react';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ScrollableMenuContent } from '@platform/ui/ScrollableMenuContent';
import { useQuery } from '@tanstack/react-query';
import { GalleryBoardDestinationGroups } from '@workbench/image-actions/GalleryBoardDestinationGroups';
import { useWidgetValuesSelector } from '@workbench/WorkbenchContext';
import { ChevronRightIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerContextAction, LayerContextBoardChoice } from './layerContextActions';

const SUBMENU_POSITIONING = { placement: 'right-start' } as const;

/**
 * A `picker: 'board'` layer action as a submenu of the boards the caller can add to, Uncategorized first. The list
 * mounts only while the submenu is open, so a layer menu never reads boards it does not show.
 */
export const LayerSaveToBoardSubmenu = ({
  action,
  onSelect,
}: {
  action: LayerContextAction;
  onSelect: (action: LayerContextAction, board: LayerContextBoardChoice) => void;
}) => {
  const { t } = useTranslation();
  const label = t(action.labelKey, { count: action.labelCount, defaultValue: action.defaultLabel });
  const handleSelect = useCallback((board: LayerContextBoardChoice) => onSelect(action, board), [action, onSelect]);

  return (
    <Menu.Root lazyMount positioning={SUBMENU_POSITIONING} unmountOnExit>
      <Menu.TriggerItem asChild>
        {/* zag's trigger item has no disabled prop: `data-disabled` keeps arrow keys from highlighting it, which would
            let ArrowRight open the submenu; the native attribute stops pointer hover and click. */}
        <button
          aria-disabled={action.isDisabled || undefined}
          aria-label={label}
          data-disabled={action.isDisabled ? '' : undefined}
          disabled={action.isDisabled}
          type="button"
        >
          <HStack gap="2" minW="0" w="full">
            <Icon as={action.icon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
            <Text flex="1" fontSize="md">
              {label}
            </Text>
            <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
          </HStack>
        </button>
      </Menu.TriggerItem>
      <Portal>
        <Menu.Positioner>
          <ScrollableMenuContent minW="13rem" py="1">
            <BoardChoices onSelect={handleSelect} />
          </ScrollableMenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const BoardChoices = ({ onSelect }: { onSelect: (board: LayerContextBoardChoice) => void }) => {
  const { t } = useTranslation();
  // The Gallery panel's own listing (same key, shared cache), so the boards read as the panel shows them.
  const galleryValues = useWidgetValuesSelector('gallery', (values) => values);
  const { data, isError, isPending, refetch } = useQuery(
    galleryBoardsOptions(getGalleryListingBoardsQuery(getGallerySettings(galleryValues)))
  );
  const boards = useHostGalleryBoards(data);
  const uncategorizedLabel = t('widgets.gallery.uncategorized');
  const handleSelectUncategorized = useCallback(
    () => onSelect({ id: 'none', label: uncategorizedLabel }),
    [onSelect, uncategorizedLabel]
  );
  const handleSelectBoard = useCallback(
    (board: GalleryBoard) => onSelect({ id: board.id, label: getGalleryBoardLabel(board, t) }),
    [onSelect, t]
  );
  const handleRetry = useCallback(() => void refetch(), [refetch]);

  return (
    <>
      <Menu.Item value="board-none" onClick={handleSelectUncategorized}>
        <MiddleTruncate fontSize="md" minW="0" text={uncategorizedLabel} />
      </Menu.Item>
      {isPending ? (
        <Menu.Item disabled value="boards-loading">
          <Text color="fg.muted" fontSize="md">
            {t('common.loading')}
          </Text>
        </Menu.Item>
      ) : isError ? (
        <Menu.Item closeOnSelect={false} value="boards-retry" onClick={handleRetry}>
          <Text fontSize="md">{t('widgets.gallery.retryLoadingBoards')}</Text>
        </Menu.Item>
      ) : null}
      <GalleryBoardDestinationGroups boards={boards} onSelect={handleSelectBoard} />
    </>
  );
};
