import type { GalleryBoard } from '@features/gallery/core/types';

import { Icon } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { isDateBoardId } from '@features/gallery/data/backend';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { UploadIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { useGalleryUploadInput } from './useGalleryUploadInput';

export const GalleryUploadButton = ({
  boards,
  selectedBoardId,
  onUploadFiles,
}: {
  boards: GalleryBoard[];
  selectedBoardId: string;
  onUploadFiles: (files: File[]) => Promise<void>;
}) => {
  const { t } = useTranslation();
  const selectedBoard = boards.find((board) => board.id === selectedBoardId);
  const isVirtualTarget = isDateBoardId(selectedBoardId);
  const label = isVirtualTarget
    ? t('widgets.gallery.uploadsUnavailableForDateBoards')
    : t('widgets.gallery.uploadMediaToBoard', {
        name: selectedBoard ? getGalleryBoardLabel(selectedBoard, t) : t('widgets.gallery.selectedBoardFallback'),
      });

  const { inputProps, openPicker } = useGalleryUploadInput(onUploadFiles);

  return (
    <>
      <input {...inputProps} />
      <Tooltip content={label}>
        <IconButton
          aria-label={label}
          color="fg.muted"
          disabled={isVirtualTarget}
          size="2xs"
          variant="ghost"
          onClick={openPicker}
        >
          <Icon as={UploadIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    </>
  );
};
