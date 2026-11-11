import type { GalleryItemKind } from '@features/gallery/contracts';
import type { ReactNode, Ref } from 'react';

import { Box, Flex } from '@chakra-ui/react';
import { classifyGalleryUpload } from '@features/gallery/contracts';
import { findDropZoneAt } from '@platform/browser/dropZones';
import { useNativeFileDrag } from '@platform/react/useNativeFileDrag';
import { DropZone } from '@platform/ui/DropZone';
import { useQueryClient } from '@tanstack/react-query';
import { DropZoneLabel } from '@workbench/widgets/shared/DropZoneLabel';
import { useDroppedFileUpload } from '@workbench/widgets/shared/useDroppedFileUpload';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export type PreviewDropZone = 'show' | 'compare';

const PREVIEW_DROP_ZONE_ATTRIBUTE = 'data-preview-drop-zone';

const findPreviewDropZone = (root: Element, point: { clientX: number; clientY: number }): PreviewDropZone | null => {
  const zone = findDropZoneAt(root, PREVIEW_DROP_ZONE_ATTRIBUTE, point);
  return zone === 'show' || zone === 'compare' ? zone : null;
};

// Only an image can be compared; a file the OS gave no type might be one.
const mayBeImage = (type: string): boolean =>
  type === '' || classifyGalleryUpload({ name: '', type })?.kind === 'image';

const PreviewDropZoneTarget = ({ isOver, label, zone }: { isOver: boolean; label: string; zone: PreviewDropZone }) => (
  <DropZone
    alignItems="center"
    data-over={isOver || undefined}
    display="flex"
    flex="1"
    isOver={isOver}
    justifyContent="center"
    minW="0"
    opacity={isOver ? 1 : 0.85}
    rounded="lg"
    variant="overlay"
    {...{ [PREVIEW_DROP_ZONE_ATTRIBUTE]: zone }}
  >
    <DropZoneLabel>{label}</DropZoneLabel>
  </DropZone>
);

/**
 * Preview's root, taking media files dragged in from the operating system. While they hover it shows where they go:
 * shown in Preview, and beside it, while Preview shows an image outside live progress and comparison, the compare zone
 * a gallery drag offers there. A drop uploads them as the Gallery does, then shows them (see `importPreviewFileDrop`,
 * loaded on the first drop to keep uploads out of Preview's startup).
 */
export const PreviewDropSurface = ({
  children,
  isComparing,
  isFollowingLive,
  ref,
  shownKind,
}: {
  children: ReactNode;
  isComparing: boolean;
  isFollowingLive: boolean;
  ref?: Ref<HTMLDivElement>;
  /** The kind of gallery item Preview shows, if any. */
  shownKind: GalleryItemKind | null;
}) => {
  const { t } = useTranslation();
  const { gallery, notifications } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const startDroppedFileUpload = useDroppedFileUpload({
    pendingKey: 'widgets.preview.dropPending',
    uploadingKey: 'widgets.preview.dropUploading',
  });
  const importDroppedFiles = useCallback(
    (files: File[], zone: PreviewDropZone) =>
      startDroppedFileUpload(files, {
        load: () => import('./importPreviewFileDrop'),
        run: ({ importPreviewFileDrop }, context) =>
          importPreviewFileDrop({ ...context, files, gallery, notifications, queries, queryClient, t, zone }),
        runFailedMessage: t('widgets.gallery.uploadFailed', { failed: files.length }),
      }),
    [gallery, notifications, queries, queryClient, startDroppedFileUpload, t]
  );
  const { drag, handlers } = useNativeFileDrag({ findZone: findPreviewDropZone, onDrop: importDroppedFiles });
  const showsCompare =
    !isFollowingLive && !isComparing && shownKind === 'image' && drag !== null && drag.fileTypes.some(mayBeImage);

  return (
    // Let the dot grid fill the widget; containerType bounds Details cqh sizing to this surface.
    <Box ref={ref} containerType="size" h="full" position="relative" w="full" {...handlers}>
      {children}
      {drag ? (
        // Nearly opaque: what Preview shows stays only as a trace, so it never reads through the zones' labels.
        <Flex bg="bg.inset/95" gap="1" inset="0" p="1" pointerEvents="none" position="absolute" zIndex="2">
          <PreviewDropZoneTarget isOver={drag.zone === 'show'} label={t('widgets.preview.dropToShow')} zone="show" />
          {showsCompare ? (
            <PreviewDropZoneTarget
              isOver={drag.zone === 'compare'}
              label={t('widgets.preview.dropToCompare')}
              zone="compare"
            />
          ) : null}
        </Flex>
      ) : null}
    </Box>
  );
};
