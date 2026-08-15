import type { GalleryBoard } from '@features/gallery/core/types';

import { Icon } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { isDateBoardId } from '@features/gallery/data/backend';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { UploadIcon } from 'lucide-react';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export const ACCEPTED_UPLOAD_EXTENSIONS = 'image/png,image/jpeg,image/webp,video/mp4,.png,.jpg,.jpeg,.webp,.mp4';
export const UPLOAD_INPUT_STYLE = { display: 'none' } as const;

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
