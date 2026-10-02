import { HStack, Icon, Text } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { getGallerySettings } from '@features/gallery/core/settings';
import { galleryBoardsOptions } from '@features/gallery/data/queries';
import { Button } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useQuery } from '@tanstack/react-query';
import { ChevronsDownUpIcon, ChevronsUpDownIcon, ImageIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryWidgetProps } from './GalleryUiContext';

import { BoardCover, BoardCoverIcon } from './GalleryBoardCover';
import { getGalleryProjectBoardId, getGallerySelectedBoardId } from './galleryStateView';
import { useGalleryUi } from './GalleryUiContext';

type GalleryChromeProps = { region: GalleryWidgetProps['region'] };

const LABEL_EXPANDED_PROPS = { bg: 'transparent' } as const;

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
  const resolvedBoards = useMemo(() => boards ?? [], [boards]);

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
  const { projectName } = useGalleryUi();
  const { boards, gallery, galleryValues, selectedBoardId, settings } = useGalleryChromeBoards();
  const selectedBoard = boards.find((board) => board.id === selectedBoardId);
  // The project board renames with its project server-side, but the fetched
  // list lags a rename; the live project name is authoritative (as in the
  // boards panel), so the header follows a rename at once.
  const boardName = !selectedBoard
    ? t('widgets.gallery.selectedBoardFallback')
    : selectedBoard.id === getGalleryProjectBoardId(galleryValues)
      ? projectName
      : getGalleryBoardLabel(selectedBoard, t);
  const isCollapsed = settings.boardPanelCollapsed;

  const toggleBoards = useCallback(
    () => gallery.updateSettings({ boardPanelCollapsed: !isCollapsed }),
    [gallery, isCollapsed]
  );

  return (
    <HStack flex="1" gap="1" minW="0" pe={region === 'center' ? undefined : '2'}>
      {region === 'center' ? null : (
        <Text flexShrink={0} fontSize="xs" fontWeight="700" pe="2">
          {t('widgets.labels.gallery')}
        </Text>
      )}
      <Text color="fg.subtle" flexShrink={0} fontSize="xs">
        /
      </Text>
      {/* No tooltip: it renders over the button and steals its hover. */}
      <Button
        _expanded={LABEL_EXPANDED_PROPS}
        aria-expanded={!isCollapsed}
        aria-label={t('widgets.gallery.toggleBoardsNamed', { name: boardName })}
        maxW="14rem"
        minW="0"
        // The cover keeps the same 2px inset on every side of the 24px button.
        ps="0.5"
        size="2xs"
        variant="ghost"
        onClick={toggleBoards}
      >
        {selectedBoard ? <BoardCover board={selectedBoard} /> : <BoardCoverIcon icon={ImageIcon} />}
        <MiddleTruncate fontWeight="600" minW="0" text={boardName} />
        <Icon as={isCollapsed ? ChevronsUpDownIcon : ChevronsDownUpIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
      </Button>
    </HStack>
  );
};
