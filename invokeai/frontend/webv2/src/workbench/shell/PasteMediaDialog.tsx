import type { GalleryBoard, GalleryImage, GalleryItem, GalleryItemKind } from '@features/gallery/contracts';
import type { GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';

import { Box, Flex, Image, Portal, SimpleGrid, Stack, Text } from '@chakra-ui/react';
import {
  classifyGalleryUpload,
  galleryImageItemToGalleryImage,
  getGalleryBoardLabel,
  getGalleryAutoAddBoardId,
  getGallerySelectedBoardId,
  getGallerySettings,
  getGalleryView,
  isGalleryImageItem,
} from '@features/gallery/contracts';
import { galleryBoardsOptions } from '@features/gallery/queries';
import { useGalleryUploadAction } from '@features/gallery/react';
import { createGenerateFormValuesSelector } from '@features/generation/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, CloseButton } from '@platform/ui';
import { Dialog } from '@platform/ui/Dialog';
import { useQuery } from '@tanstack/react-query';
import {
  getGalleryCanvasImportMenuItems,
  useDeletionConfirmation,
  useImageActions,
  type ImageActions,
} from '@workbench/image-actions';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectId, useActiveProjectSelector, useWidgetValuesSelector } from '@workbench/WorkbenchContext';
import { FilmIcon, ImagePlusIcon, ImagesIcon } from 'lucide-react';
import { useCallback, useId, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { PasteMediaRequest } from './PasteMediaRuntime';

type PasteMediaChoice =
  | { kind: 'gallery' }
  | { kind: 'canvas'; destination: GalleryCanvasImportDestination }
  | { kind: 'reference' };

const selectGenerateValues = createGenerateFormValuesSelector();
const CANVAS_ITEMS = getGalleryCanvasImportMenuItems(true);
const PREVIEW_LIMIT = 6;
const EMPTY_BOARDS: GalleryBoard[] = [];

/** Upload and follow-on actions report their own failures. */
const routePastedFiles = async (
  files: File[],
  choice: PasteMediaChoice,
  {
    addReferenceImage,
    sendToCanvas,
    uploadFiles,
  }: {
    addReferenceImage: ImageActions['useAsReferenceImage'];
    sendToCanvas: ImageActions['sendToCanvas'];
    uploadFiles: (files: File[]) => Promise<GalleryItem[]>;
  }
): Promise<void> => {
  const items = await uploadFiles(files);
  if (choice.kind === 'gallery') {
    return;
  }
  const images: GalleryImage[] = items.filter(isGalleryImageItem).map(galleryImageItemToGalleryImage);
  if (images.length === 0) {
    return;
  }
  if (choice.kind === 'canvas') {
    await sendToCanvas(images, choice.destination);
    return;
  }
  for (const image of images) {
    addReferenceImage(image);
  }
};

const CanvasDestinationButton = ({
  destination,
  label,
  onChoose,
}: {
  destination: GalleryCanvasImportDestination;
  label: string;
  onChoose: (destination: GalleryCanvasImportDestination) => void;
}) => {
  const handleClick = useCallback(() => onChoose(destination), [destination, onChoose]);

  return (
    <Button justifyContent="flex-start" size="lg" variant="outline" onClick={handleClick}>
      {label}
    </Button>
  );
};

/**
 * Upload pasted media to the current board before forwarding it to canvas/reference actions so assets remain
 * findable.
 */
export const PasteMediaDialog = ({
  isOpen,
  onExitComplete,
  request,
}: {
  isOpen: boolean;
  onExitComplete: () => void;
  request: PasteMediaRequest;
}) => {
  const { t } = useTranslation();
  const projectId = useActiveProjectId();
  const galleryValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'gallery'));
  const generateValues = useWidgetValuesSelector('generate', selectGenerateValues);
  // The gallery's own boards query (same key, shared cache): the board the
  // person is looking at may be archived or a date bucket, which the default
  // listing omits.
  const gallerySettings = getGallerySettings(galleryValues);
  const boards =
    useQuery(
      galleryBoardsOptions({
        includeArchived: gallerySettings.showArchivedBoards,
        includeDateBoards: gallerySettings.showDateBoards,
        orderBy: gallerySettings.boardOrderBy,
        orderDir: gallerySettings.boardOrderDir,
      })
    ).data ?? EMPTY_BOARDS;
  // Uploads land where generation results do: the auto-add board — never a
  // date bucket, which cannot hold items.
  const destinationBoardId = getGalleryAutoAddBoardId(galleryValues) ?? 'none';
  const selectedBoardId = getGallerySelectedBoardId(galleryValues, boards);
  const galleryView = getGalleryView(galleryValues);
  const getCurrentGalleryLocation = useCallback(
    () => ({ galleryView, selectedBoardId }),
    [galleryView, selectedBoardId]
  );
  const uploadFiles = useGalleryUploadAction({
    boards,
    getCurrentGalleryLocation,
    selectedBoardId: destinationBoardId,
  });
  const { dialog: deletionDialog, requestDeletionConfirmation } = useDeletionConfirmation();
  const actions = useImageActions({ boards, generateValues, projectId, requestDeletionConfirmation });
  const addReferenceImage = actions.useAsReferenceImage;

  const [previews, setPreviews] = useState<{ kind: GalleryItemKind; url: string }[]>([]);
  // Object URLs are created and revoked together per mount, so StrictMode's
  // simulated remount cannot revoke URLs a live <img> still points at.
  useMountEffect(() => {
    const next = request.files.slice(0, PREVIEW_LIMIT).map((file) => ({
      kind: classifyGalleryUpload(file)?.kind ?? 'image',
      url: URL.createObjectURL(file),
    }));
    setPreviews(next);
    return () => next.forEach((preview) => URL.revokeObjectURL(preview.url));
  });
  const hasImages = request.files.some((file) => classifyGalleryUpload(file)?.kind === 'image');
  const hasVideos = request.files.some((file) => classifyGalleryUpload(file)?.kind === 'video');
  const destinationBoard = boards.find((candidate) => candidate.id === destinationBoardId);
  const boardName =
    destinationBoardId === 'none'
      ? t('widgets.gallery.uncategorized')
      : destinationBoard
        ? getGalleryBoardLabel(destinationBoard, t)
        : null;

  const canvasHeadingId = useId();
  const galleryButtonRef = useRef<HTMLButtonElement | null>(null);
  const getInitialFocusEl = useCallback(() => galleryButtonRef.current, []);
  const returnFocus = useCallback(
    () => (request.returnFocus?.isConnected ? request.returnFocus : null),
    [request.returnFocus]
  );
  const close = request.settle;
  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (!open) {
        close();
      }
    },
    [close]
  );
  const choose = useCallback(
    (choice: PasteMediaChoice) => {
      close();
      void routePastedFiles(request.files, choice, {
        addReferenceImage,
        sendToCanvas: actions.sendToCanvas,
        uploadFiles,
      });
    },
    [actions.sendToCanvas, addReferenceImage, close, request.files, uploadFiles]
  );
  const chooseGallery = useCallback(() => choose({ kind: 'gallery' }), [choose]);
  const chooseReference = useCallback(() => choose({ kind: 'reference' }), [choose]);
  const chooseCanvas = useCallback(
    (destination: GalleryCanvasImportDestination) => choose({ destination, kind: 'canvas' }),
    [choose]
  );

  return (
    <>
      <Dialog.Root
        open={isOpen}
        finalFocusEl={returnFocus}
        initialFocusEl={getInitialFocusEl}
        placement="center"
        size="sm"
        onExitComplete={onExitComplete}
        onOpenChange={handleOpenChange}
      >
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title>{t('shell.pasteMedia.title', { count: request.files.length })}</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="3">
                  <Flex gap="2" wrap="wrap">
                    {previews.map((preview, index) => (
                      <Box
                        key={preview.url}
                        bg="bg.inset"
                        borderRadius="control"
                        borderWidth="1px"
                        boxSize="64px"
                        overflow="hidden"
                      >
                        {preview.kind === 'video' ? (
                          <Flex align="center" color="fg.muted" h="full" justify="center">
                            <FilmIcon aria-label={t('shell.pasteMedia.videoPreview', { index: index + 1 })} size={20} />
                          </Flex>
                        ) : (
                          <Image
                            alt={t('shell.pasteMedia.imagePreview', { index: index + 1 })}
                            h="full"
                            objectFit="cover"
                            src={preview.url}
                            w="full"
                          />
                        )}
                      </Box>
                    ))}
                    {request.files.length > PREVIEW_LIMIT ? (
                      <Flex align="center" color="fg.muted" fontSize="lg" h="64px" px="2">
                        {t('shell.pasteMedia.morePreviews', { count: request.files.length - PREVIEW_LIMIT })}
                      </Flex>
                    ) : null}
                  </Flex>
                  <Text color="fg.muted" fontSize="lg">
                    {t('shell.pasteMedia.description', { count: request.files.length })}
                  </Text>
                  <Stack gap="2">
                    <Button
                      ref={galleryButtonRef}
                      justifyContent="flex-start"
                      size="lg"
                      variant="outline"
                      onClick={chooseGallery}
                    >
                      <ImagesIcon />
                      {t('shell.pasteMedia.gallery')}
                      {boardName ? (
                        <Text as="span" color="fg.muted" fontWeight="normal" truncate>
                          {boardName}
                        </Text>
                      ) : null}
                    </Button>
                    {hasImages ? (
                      <>
                        <Text id={canvasHeadingId} color="fg.muted" fontSize="md" mt="1">
                          {t('shell.pasteMedia.canvasHeading')}
                        </Text>
                        <SimpleGrid aria-labelledby={canvasHeadingId} columns={2} gap="2" role="group">
                          {CANVAS_ITEMS.map((item) => (
                            <CanvasDestinationButton
                              key={item.destination}
                              destination={item.destination}
                              label={t(item.label)}
                              onChoose={chooseCanvas}
                            />
                          ))}
                        </SimpleGrid>
                        {actions.canUseAsReferenceImage ? (
                          <Button
                            justifyContent="flex-start"
                            mt="1"
                            size="lg"
                            variant="outline"
                            onClick={chooseReference}
                          >
                            <ImagePlusIcon />
                            {t('shell.pasteMedia.reference')}
                          </Button>
                        ) : null}
                      </>
                    ) : null}
                    {hasImages && hasVideos ? (
                      <Text color="fg.muted" fontSize="md">
                        {t('shell.pasteMedia.videosGalleryOnly')}
                      </Text>
                    ) : null}
                  </Stack>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer>
                <Button color="fg" size="lg" variant="ghost" onClick={close}>
                  {t('common.cancel')}
                </Button>
              </Dialog.Footer>
              <Dialog.CloseTrigger asChild>
                <CloseButton aria-label={t('common.close')} />
              </Dialog.CloseTrigger>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
      {deletionDialog}
    </>
  );
};
