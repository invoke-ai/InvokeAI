import type { GalleryBoard } from '@features/gallery/core/types';
import type { TFunction } from 'i18next';

import { Icon } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { isDateBoardId } from '@features/gallery/data/backend';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { UploadIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { useGalleryUploadInput } from './useGalleryUploadInput';

/** Tooltip/aria text for an upload control; null when the board cannot receive uploads. */
export const getGalleryUploadTargetLabel = (
  boards: readonly GalleryBoard[],
  selectedBoardId: string,
  t: TFunction
): { isAvailable: boolean; label: string } => {
  if (isDateBoardId(selectedBoardId)) {
    return { isAvailable: false, label: t('widgets.gallery.uploadsUnavailableForDateBoards') };
  }

  const selectedBoard = boards.find((board) => board.id === selectedBoardId);

  return {
    isAvailable: true,
    label: t('widgets.gallery.uploadMediaToBoard', {
      name: selectedBoard ? getGalleryBoardLabel(selectedBoard, t) : t('widgets.gallery.selectedBoardFallback'),
    }),
  };
};

export const GalleryUploadButton = ({
  boards,
  selectedBoardId,
  onUploadFiles,
}: {
  boards: GalleryBoard[];
  selectedBoardId: string;
  onUploadFiles: (files: File[]) => Promise<unknown>;
}) => {
  const { t } = useTranslation();
  const { isAvailable, label } = getGalleryUploadTargetLabel(boards, selectedBoardId, t);

  const { inputProps, openPicker } = useGalleryUploadInput(onUploadFiles);

  return (
    <>
      <input {...inputProps} />
      <Tooltip content={label}>
        <IconButton
          aria-label={label}
          color="fg.muted"
          disabled={!isAvailable}
          size="xs"
          variant="ghost"
          onClick={openPicker}
        >
          <Icon as={UploadIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    </>
  );
};
