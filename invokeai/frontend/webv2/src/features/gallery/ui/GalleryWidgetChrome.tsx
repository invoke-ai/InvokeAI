import type { GalleryBoard } from '@features/gallery/core/types';

import { HStack, Icon, Text } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { getGallerySettings } from '@features/gallery/core/settings';
import { isDateBoardId } from '@features/gallery/data/backend';
import { galleryBoardsOptions } from '@features/gallery/data/queries';
import { Button, IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { useQuery } from '@tanstack/react-query';
import { ChevronsDownUpIcon, ChevronsUpDownIcon, UploadIcon } from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryWidgetProps } from './GalleryUiContext';

import { GallerySettingsMenu } from './GallerySettingsMenu';
import { getGallerySelectedBoardId, getGalleryView } from './galleryStateView';
import { useGalleryUi } from './GalleryUiContext';
import { useGalleryUploadAction } from './useGalleryUploadAction';

type GalleryChromeProps = { region: GalleryWidgetProps['region'] };

const ACCEPTED_UPLOAD_EXTENSIONS = 'image/png,image/jpeg,image/webp,video/mp4,.png,.jpg,.jpeg,.webp,.mp4';
const UPLOAD_INPUT_STYLE = { display: 'none' } as const;
// Ghost buttons fill when expanded, and the boards start expanded — which left
// the label looking pressed at rest. The chevron carries the state instead.
const LABEL_EXPANDED_PROPS = { bg: 'transparent' } as const;

/**
 * Frame chrome mounts outside `GalleryWidgetView` and so outside its context;
 * it reads the App-owned UI port directly. The boards query is the one the view
 * already caches.
 */
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

/** Separate from the read-only hook so the label skips assembling upload actions. */
const useGalleryChromeActions = () => {
  const { boards, galleryValues, selectedBoardId, ...rest } = useGalleryChromeBoards();
  const galleryView = getGalleryView(galleryValues);
  const currentGalleryLocationRef = useRef({ galleryView, selectedBoardId });

  // eslint-disable-next-line react/react-compiler
  currentGalleryLocationRef.current = { galleryView, selectedBoardId };
  const getCurrentGalleryLocation = useCallback(() => currentGalleryLocationRef.current, []);
  const uploadFiles = useGalleryUploadAction({ boards, getCurrentGalleryLocation, selectedBoardId });

  return { ...rest, boards, galleryValues, selectedBoardId, uploadFiles };
};

/**
 * `Gallery / [Board]`, where the board name expands and collapses the panel —
 * the shape the workflow widget uses for its library trigger.
 */
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
    // End padding keeps the name off the frame's actions; the center floats
    // this in its own island and is exempt.
    <HStack flex="1" gap="1" minW="0" pe={region === 'center' ? undefined : '2'}>
      {/* The center's view selector already names the widget. */}
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
        <Text fontWeight="600" minW="0" truncate>
          {boardName}
        </Text>
        <Icon as={isCollapsed ? ChevronsUpDownIcon : ChevronsDownUpIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
      </Button>
    </HStack>
  );
};

/**
 * Upload and grid settings, in every region. The frame owns these because the
 * center hoists header chrome into an island that would otherwise sit empty.
 */
export const GalleryWidgetHeaderActions = (_props: GalleryChromeProps) => {
  const { boards, gallery, selectedBoardId, settings, uploadFiles } = useGalleryChromeActions();

  return (
    <HStack gap="0.5">
      <GalleryUploadButton boards={boards} selectedBoardId={selectedBoardId} onUploadFiles={uploadFiles} />
      <GallerySettingsMenu settings={settings} onUpdateSettings={gallery.updateSettings} />
    </HStack>
  );
};

const GalleryUploadButton = ({
  boards,
  selectedBoardId,
  onUploadFiles,
}: {
  boards: GalleryBoard[];
  selectedBoardId: string;
  onUploadFiles: (files: File[]) => Promise<void>;
}) => {
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedBoard = boards.find((board) => board.id === selectedBoardId);
  const isVirtualTarget = isDateBoardId(selectedBoardId);
  const label = isVirtualTarget
    ? t('widgets.gallery.uploadsUnavailableForDateBoards')
    : t('widgets.gallery.uploadMediaToBoard', {
        name: selectedBoard ? getGalleryBoardLabel(selectedBoard, t) : t('widgets.gallery.selectedBoardFallback'),
      });

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.currentTarget.files ?? []);

      event.currentTarget.value = '';

      if (files.length > 0) {
        void onUploadFiles(files);
      }
    },
    [onUploadFiles]
  );

  const handleUploadClick = useCallback(() => fileInputRef.current?.click(), []);

  return (
    <>
      <input
        accept={ACCEPTED_UPLOAD_EXTENSIONS}
        multiple
        ref={fileInputRef}
        style={UPLOAD_INPUT_STYLE}
        type="file"
        onChange={handleFileChange}
      />
      <Tooltip content={label}>
        <IconButton
          aria-label={label}
          color="fg.muted"
          disabled={isVirtualTarget}
          size="2xs"
          variant="ghost"
          onClick={handleUploadClick}
        >
          <Icon as={UploadIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    </>
  );
};
