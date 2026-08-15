import { HStack, Icon, Text } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { getGallerySettings } from '@features/gallery/core/settings';
import { galleryBoardsOptions } from '@features/gallery/data/queries';
import { Button } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useQuery } from '@tanstack/react-query';
import { ChevronsDownUpIcon, ChevronsUpDownIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryWidgetProps } from './GalleryUiContext';

import { GallerySettingsMenu } from './GallerySettingsMenu';
import { getGallerySelectedBoardId } from './galleryStateView';
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
  const { boards, gallery, selectedBoardId, settings } = useGalleryChromeBoards();
  const selectedBoard = boards.find((board) => board.id === selectedBoardId);
  const boardName = selectedBoard ? getGalleryBoardLabel(selectedBoard, t) : t('widgets.gallery.selectedBoardFallback');
  const isCollapsed = settings.boardPanelCollapsed;

  const toggleBoards = useCallback(
    () => gallery.updateSettings({ boardPanelCollapsed: !isCollapsed }),
    [gallery, isCollapsed]
  );

  return (
    <HStack flex="1" gap="1" minW="0" pe={region === 'center' ? undefined : '2'}>
      {region === 'center' ? null : (
        <Text flexShrink={0} fontSize="xs" fontWeight="700">
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
        size="2xs"
        variant="ghost"
        onClick={toggleBoards}
      >
        <MiddleTruncate fontWeight="600" minW="0" text={boardName} />
        <Icon as={isCollapsed ? ChevronsUpDownIcon : ChevronsDownUpIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
      </Button>
    </HStack>
  );
};

export const GalleryWidgetHeaderActions = (_props: GalleryChromeProps) => {
  const { gallery, settings } = useGalleryChromeBoards();

  return (
    <HStack gap="0.5">
      <GallerySettingsMenu settings={settings} onUpdateSettings={gallery.updateSettings} />
    </HStack>
  );
};
