/* eslint-disable react/react-compiler */
import type { GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';
import type { GalleryBoard, GalleryImage } from '@workbench/gallery/api';

import { Dialog, HStack, Icon, Menu, Portal, ScrollArea, Text } from '@chakra-ui/react';
import { Button, MenuContent, Tooltip } from '@workbench/components/ui';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import {
  AsteriskIcon,
  ChevronRightIcon,
  CopyIcon,
  DownloadIcon,
  ExternalLinkIcon,
  EyeIcon,
  FileImageIcon,
  FolderIcon,
  ImageIcon,
  ImagesIcon,
  LayersIcon,
  QuoteIcon,
  RulerIcon,
  ScanIcon,
  ScissorsIcon,
  ShuffleIcon,
  SproutIcon,
  StarIcon,
  Trash2Icon,
  TypeIcon,
  WorkflowIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import type { ImageActions } from './useImageActions';

import { EMPTY_IMAGE_RECALL_CAPABILITIES, type ImageRecallCapabilities } from './imageRecall';

export interface ImageContextMenuTarget {
  /** Right-clicked image first; more entries switch the menu into bulk mode. */
  images: GalleryImage[];
  x: number;
  y: number;
}

/** Extras on top of the shared MenuContent surface styling. */
const MENU_CONTENT_PROPS = {
  minW: '13rem',
  overflow: 'hidden',
} as const;
const MENU_POSITIONING_PROPS = { placement: 'right-start' } as const;
const QUICK_MENU_TOOLTIP_CONTENT_PROPS = { fontSize: '2xs' } as const;
const QUICK_MENU_TOOLTIP_POSITIONING_PROPS = { placement: 'top' } as const;

const isUsableGalleryImage = (value: unknown): value is GalleryImage =>
  Boolean(value) &&
  typeof value === 'object' &&
  typeof (value as GalleryImage).imageName === 'string' &&
  typeof (value as GalleryImage).imageUrl === 'string' &&
  typeof (value as GalleryImage).thumbnailUrl === 'string';

interface GalleryCanvasImportMenuItem {
  destination: GalleryCanvasImportDestination;
  label: string;
  value: string;
}

const GALLERY_CANVAS_IMPORT_MENU_ITEMS = [
  { destination: 'raster', label: 'widgets.canvas.import.raster' },
  { destination: 'control', label: 'widgets.canvas.import.control' },
  { destination: 'inpaint-mask', label: 'widgets.canvas.import.inpaintMask' },
  { destination: 'regional-guidance', label: 'widgets.canvas.import.regionalGuidance' },
  { destination: 'regional-reference', label: 'widgets.canvas.import.regionalReference' },
  { destination: 'control-resized', label: 'widgets.canvas.import.resizedControl' },
] as const satisfies readonly Omit<GalleryCanvasImportMenuItem, 'value'>[];

export const getGalleryCanvasImportMenuItems = (isBulk: boolean): GalleryCanvasImportMenuItem[] =>
  GALLERY_CANVAS_IMPORT_MENU_ITEMS.map((item) => ({
    ...item,
    value: `send-${isBulk ? 'images' : 'image'}-to-canvas-${item.destination}`,
  }));

export const getImageContextMenuImages = (target: ImageContextMenuTarget | null): GalleryImage[] => {
  if (!Array.isArray(target?.images)) {
    return [];
  }

  return target.images.filter(isUsableGalleryImage);
};

export const getImageContextMenuRecallRequestKey = (image: GalleryImage | null, isBulk: boolean): string | null => {
  if (!image || isBulk) {
    return null;
  }

  return `${image.imageName}:${image.width}:${image.height}`;
};

/**
 * Shared right-click menu for backend images, usable from any widget (gallery
 * grid, preview, ...). Anchored to the cursor through a virtual rect, so it
 * needs no trigger element — set `target` to open it.
 */
export const ImageContextMenu = ({
  actions,
  boards,
  target,
  onClose,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  target: ImageContextMenuTarget | null;
  onClose: () => void;
}) => {
  const images = getImageContextMenuImages(target);
  const image = images[0] ?? null;

  if (!target || !image) {
    return null;
  }

  return (
    <ImageContextMenuContent
      key={`${image.imageName}:${images.length}`}
      actions={actions}
      boards={boards}
      image={image}
      images={images}
      target={target}
      onClose={onClose}
    />
  );
};

const ImageContextMenuContent = ({
  actions,
  boards,
  image,
  images,
  target,
  onClose,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  image: GalleryImage;
  images: GalleryImage[];
  target: ImageContextMenuTarget;
  onClose: () => void;
}) => {
  const confirmImageDeletion = useWorkbenchPreferenceSelector((preferences) => preferences.confirmImageDeletion);
  const [pendingDeletion, setPendingDeletion] = useState<string[] | null>(null);
  const [recallCapabilities, setRecallCapabilities] = useState<ImageRecallCapabilities>(
    EMPTY_IMAGE_RECALL_CAPABILITIES
  );
  const [isLoadingRecallCapabilities, setIsLoadingRecallCapabilities] = useState(false);
  const targetRef = useRef(target);
  const imageRef = useRef<GalleryImage | null>(null);

  targetRef.current = target;

  const isBulk = images.length > 1;
  const imageNames = useMemo(() => images.map((candidate) => candidate.imageName), [images]);
  const recallRequestKey = getImageContextMenuRecallRequestKey(image, isBulk);
  const getImageRecallCapabilities = actions.getImageRecallCapabilities;

  imageRef.current = image;

  useEffect(() => {
    if (!recallRequestKey) {
      return;
    }

    let isCancelled = false;
    const recallImage = imageRef.current;

    if (!recallImage) {
      return;
    }

    setIsLoadingRecallCapabilities((current) => (current ? current : true));
    getImageRecallCapabilities(recallImage)
      .then((capabilities) => {
        if (!isCancelled) {
          setRecallCapabilities(capabilities);
        }
      })
      .catch(() => {
        if (!isCancelled) {
          setRecallCapabilities(EMPTY_IMAGE_RECALL_CAPABILITIES);
        }
      })
      .finally(() => {
        if (!isCancelled) {
          setIsLoadingRecallCapabilities(false);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [getImageRecallCapabilities, recallRequestKey]);

  const requestDeletion = useCallback(
    (names: string[]) => {
      if (confirmImageDeletion) {
        setPendingDeletion(names);
      } else {
        void actions.deleteImages(names);
      }
    },
    [actions, confirmImageDeletion]
  );

  const positioning = useMemo(
    () => ({
      getAnchorRect: () => {
        const currentTarget = targetRef.current;

        return { height: 1, width: 1, x: currentTarget.x, y: currentTarget.y };
      },
      placement: 'bottom-start' as const,
    }),
    []
  );
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );
  const handleDeleteDialogOpenChange = useCallback((event: { open: boolean }) => {
    if (!event.open) {
      setPendingDeletion(null);
    }
  }, []);
  const handleCancelDeletion = useCallback(() => setPendingDeletion(null), []);
  const handleConfirmDeletion = useCallback(() => {
    if (pendingDeletion) {
      void actions.deleteImages(pendingDeletion);
    }

    setPendingDeletion(null);
  }, [actions, pendingDeletion]);

  return (
    <>
      <Menu.Root lazyMount open positioning={positioning} unmountOnExit onOpenChange={handleOpenChange}>
        <Portal>
          <Menu.Positioner>
            <MenuContent {...MENU_CONTENT_PROPS} maxH="min(28rem, calc(100vh - 2rem))" minW="16rem" overflowY="auto">
              {isBulk ? (
                <BulkMenuItems
                  actions={actions}
                  boards={boards}
                  imageNames={imageNames}
                  images={images}
                  onRequestDeletion={requestDeletion}
                />
              ) : (
                <SingleImageMenuItems
                  actions={actions}
                  boards={boards}
                  image={image}
                  images={images}
                  isLoadingRecallCapabilities={isLoadingRecallCapabilities}
                  onRequestDeletion={requestDeletion}
                  recallCapabilities={recallCapabilities}
                />
              )}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Dialog.Root open={pendingDeletion !== null} role="alertdialog" onOpenChange={handleDeleteDialogOpenChange}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title fontSize="sm">
                  {pendingDeletion && pendingDeletion.length > 1
                    ? `Delete ${pendingDeletion.length} images?`
                    : 'Delete image?'}
                </Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Text color="fg.subtle" fontSize="xs">
                  This permanently deletes{' '}
                  {pendingDeletion && pendingDeletion.length > 1 ? 'these images' : 'the image'} from every board and
                  from disk. This cannot be undone. You can disable this confirmation in Settings.
                </Text>
              </Dialog.Body>
              <Dialog.Footer gap="2">
                <Button size="xs" variant="outline" onClick={handleCancelDeletion}>
                  Cancel
                </Button>
                <Button colorPalette="red" size="xs" onClick={handleConfirmDeletion}>
                  Delete
                </Button>
              </Dialog.Footer>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </>
  );
};

const SingleImageMenuItems = ({
  actions,
  boards,
  image,
  images,
  isLoadingRecallCapabilities,
  onRequestDeletion,
  recallCapabilities,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  image: GalleryImage;
  images: GalleryImage[];
  isLoadingRecallCapabilities: boolean;
  onRequestDeletion: (imageNames: string[]) => void;
  recallCapabilities: ImageRecallCapabilities;
}) => {
  const handleMove = useCallback(
    (boardId: string) => void actions.moveImagesToBoard([image.imageName], boardId),
    [actions, image.imageName]
  );
  const handleDelete = useCallback(() => onRequestDeletion([image.imageName]), [image.imageName, onRequestDeletion]);
  const handleRecallAll = useRecallImageDataHandler(actions, image, 'all');
  const handleRecallRemix = useRecallImageDataHandler(actions, image, 'remix');
  const handleRecallPrompts = useRecallImageDataHandler(actions, image, 'prompts');
  const handleRecallSeed = useRecallImageDataHandler(actions, image, 'seed');
  const handleRecallDimensions = useRecallImageDataHandler(actions, image, 'dimensions');
  const handleRecallClipSkip = useRecallImageDataHandler(actions, image, 'clipSkip');
  const handleSelectForCompare = useSelectForCompareHandler(actions, image);
  const handleUseAsReferenceImage = useUseAsReferenceImageHandler(actions, image);
  const dispatch = useWorkbenchDispatch();
  const openWidget = useOpenWorkbenchWidget();
  const handleSendToUpscale = useCallback(() => {
    openWidget('upscale', { preferredRegions: ['left'] });
    dispatch({
      type: 'patchWidgetValues',
      values: { inputImage: { height: image.height, image_name: image.imageName, width: image.width } },
      widgetId: 'upscale',
    });
    dispatch({ sourceId: 'upscale', type: 'setInvocationSource' });
  }, [dispatch, image.height, image.imageName, image.width, openWidget]);

  return (
    <>
      <HStack gap="1">
        <OpenInNewTabQuickMenuItem image={image} />
        <CopyQuickMenuItem actions={actions} image={image} />
        <DownloadQuickMenuItem actions={actions} image={image} />
        <OpenPreviewQuickMenuItem actions={actions} image={image} />
        <ToggleStarQuickMenuItem actions={actions} image={image} />
      </HStack>
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem disabled icon={WorkflowIcon} label="Load Workflow" value="load-workflow" />
      <ContextSubMenu icon={AsteriskIcon} label="Recall Metadata">
        <ContextMenuItem
          disabled={isLoadingRecallCapabilities || !recallCapabilities.all}
          icon={AsteriskIcon}
          label="Recall All"
          value="recall-all"
          onClick={handleRecallAll}
        />
        <ContextMenuItem
          disabled={isLoadingRecallCapabilities || !recallCapabilities.remix}
          icon={ShuffleIcon}
          label="Remix Image"
          value="remix"
          onClick={handleRecallRemix}
        />
        <ContextMenuItem
          disabled={isLoadingRecallCapabilities || !recallCapabilities.prompts}
          icon={QuoteIcon}
          label="Use Prompt"
          value="use-prompt"
          onClick={handleRecallPrompts}
        />
        <ContextMenuItem
          disabled={isLoadingRecallCapabilities || !recallCapabilities.seed}
          icon={SproutIcon}
          label="Use Seed"
          value="use-seed"
          onClick={handleRecallSeed}
        />
        <ContextMenuItem
          disabled={isLoadingRecallCapabilities || !recallCapabilities.dimensions}
          icon={RulerIcon}
          label="Use Size"
          value="use-size"
          onClick={handleRecallDimensions}
        />
        <ContextMenuItem
          disabled={isLoadingRecallCapabilities || !recallCapabilities.clipSkip}
          icon={ScissorsIcon}
          label="Use CLIP Skip"
          value="use-clip-skip"
          onClick={handleRecallClipSkip}
        />
      </ContextSubMenu>
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem icon={ScanIcon} label="Send to Upscale" value="send-to-upscale" onClick={handleSendToUpscale} />
      <ContextMenuItem
        disabled={!actions.canUseAsReferenceImage}
        icon={ImageIcon}
        label="Use as Reference Image"
        value="use-as-reference-image"
        onClick={handleUseAsReferenceImage}
      />
      <ContextMenuItem disabled icon={TypeIcon} label="Use as Prompt Template" value="use-as-prompt-template" />
      <ContextMenuItem
        icon={ImagesIcon}
        label="Select for Compare"
        value="select-for-compare"
        onClick={handleSelectForCompare}
      />
      <Menu.Separator borderColor="border.subtle" />
      <NewFromImageSubMenu actions={actions} images={images} isBulk={false} />
      <ChangeBoardSubMenu boards={boards} currentBoardId={image.boardId} onMove={handleMove} />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        color="fg.error"
        icon={Trash2Icon}
        label="Delete Image"
        value="delete-image"
        onClick={handleDelete}
      />
    </>
  );
};

type RecallImageDataKind = Parameters<ImageActions['recallImageData']>[1];

const useRecallImageDataHandler = (actions: ImageActions, image: GalleryImage, kind: RecallImageDataKind) =>
  useCallback(() => void actions.recallImageData(image, kind), [actions, image, kind]);

const useSelectForCompareHandler = (actions: ImageActions, image: GalleryImage) =>
  useCallback(() => actions.selectForCompare(image), [actions, image]);

const useUseAsReferenceImageHandler = (actions: ImageActions, image: GalleryImage) =>
  useCallback(() => actions.useAsReferenceImage(image), [actions, image]);

const OpenInNewTabQuickMenuItem = ({ image }: { image: GalleryImage }) => {
  const handleClick = useCallback(() => window.open(image.imageUrl, '_blank', 'noopener'), [image.imageUrl]);

  return (
    <QuickMenuItem icon={ExternalLinkIcon} label="Open in new tab" value="open-in-new-tab" onClick={handleClick} />
  );
};

const CopyQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const handleClick = useCallback(() => void actions.copyImage(image), [actions, image]);

  return <QuickMenuItem icon={CopyIcon} label="Copy to clipboard" value="copy-to-clipboard" onClick={handleClick} />;
};

const DownloadQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const handleClick = useCallback(() => void actions.downloadImage(image), [actions, image]);

  return <QuickMenuItem icon={DownloadIcon} label="Download image" value="download-image" onClick={handleClick} />;
};

const OpenPreviewQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const handleClick = useCallback(() => actions.openImageInPreview(image), [actions, image]);

  return <QuickMenuItem icon={EyeIcon} label="Open in preview" value="open-in-preview" onClick={handleClick} />;
};

const ToggleStarQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const handleClick = useCallback(
    () => void actions.setImagesStarred([image.imageName], !image.starred),
    [actions, image.imageName, image.starred]
  );

  return (
    <QuickMenuItem
      icon={StarIcon}
      label={image.starred ? 'Unstar image' : 'Star image'}
      value="toggle-starred"
      onClick={handleClick}
    />
  );
};

const BulkMenuItems = ({
  actions,
  boards,
  imageNames,
  images,
  onRequestDeletion,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  imageNames: string[];
  images: GalleryImage[];
  onRequestDeletion: (imageNames: string[]) => void;
}) => {
  const allStarred = images.every((image) => image.starred);
  const handleToggleStarred = useCallback(
    () => void actions.setImagesStarred(imageNames, !allStarred),
    [actions, allStarred, imageNames]
  );
  const handleDownload = useCallback(() => void actions.downloadImages(imageNames), [actions, imageNames]);
  const handleMove = useCallback(
    (boardId: string) => void actions.moveImagesToBoard(imageNames, boardId),
    [actions, imageNames]
  );
  const handleDelete = useCallback(() => onRequestDeletion(imageNames), [imageNames, onRequestDeletion]);

  return (
    <>
      <Text color="fg.subtle" fontSize="2xs" fontWeight="700" px="3" py="1.5" textTransform="uppercase">
        {images.length} images selected
      </Text>
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        icon={StarIcon}
        label={allStarred ? 'Unstar All' : 'Star All'}
        value="toggle-starred-all"
        onClick={handleToggleStarred}
      />
      <ContextMenuItem
        icon={DownloadIcon}
        label="Download Selection"
        value="download-selection"
        onClick={handleDownload}
      />
      <NewFromImageSubMenu actions={actions} images={images} isBulk />
      <ChangeBoardSubMenu boards={boards} currentBoardId={null} onMove={handleMove} />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        color="fg.error"
        icon={Trash2Icon}
        label="Delete Selection"
        value="delete-selection"
        onClick={handleDelete}
      />
    </>
  );
};

const NewFromImageSubMenu = ({
  actions,
  images,
  isBulk,
}: {
  actions: ImageActions;
  images: GalleryImage[];
  isBulk: boolean;
}) => (
  <ContextSubMenu icon={FileImageIcon} label={isBulk ? 'New from Images' : 'New from Image'}>
    <ContextMenuItem disabled icon={FileImageIcon} label="New Canvas from Image" value="new-canvas-from-image" />
    <GalleryCanvasImportSubMenu actions={actions} images={images} isBulk={isBulk} />
  </ContextSubMenu>
);

const GalleryCanvasImportSubMenu = ({
  actions,
  images,
  isBulk,
}: {
  actions: ImageActions;
  images: GalleryImage[];
  isBulk: boolean;
}) => {
  const { t } = useTranslation();
  const items = getGalleryCanvasImportMenuItems(isBulk);

  return (
    <ContextSubMenu icon={LayersIcon} label={t('widgets.canvas.import.newLayerFromImage')}>
      {items.map((item) => (
        <GalleryCanvasImportDestinationMenuItem key={item.destination} actions={actions} images={images} item={item} />
      ))}
    </ContextSubMenu>
  );
};

const GalleryCanvasImportDestinationMenuItem = ({
  actions,
  images,
  item,
}: {
  actions: ImageActions;
  images: GalleryImage[];
  item: GalleryCanvasImportMenuItem;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(
    () => void actions.sendToCanvas(images, item.destination),
    [actions, images, item.destination]
  );

  return <ContextMenuItem icon={LayersIcon} label={t(item.label)} value={item.value} onClick={handleClick} />;
};

const ChangeBoardSubMenu = ({
  boards,
  currentBoardId,
  onMove,
}: {
  boards: GalleryBoard[];
  currentBoardId: string | null;
  onMove: (boardId: string) => void;
}) => {
  const handleRemoveFromBoard = useCallback(() => onMove('none'), [onMove]);
  const visibleBoards = useMemo(
    () => boards.filter((board) => board.kind === 'board' && board.id !== currentBoardId),
    [boards, currentBoardId]
  );

  return (
    <ContextSubMenu icon={FolderIcon} label="Change Board" scrollArea>
      {currentBoardId !== 'none' && (
        <ContextMenuItem
          icon={FolderIcon}
          label="Remove from Board"
          value="remove-from-board"
          onClick={handleRemoveFromBoard}
        />
      )}
      {visibleBoards.map((board) => (
        <ChangeBoardMenuItem key={board.id} board={board} onMove={onMove} />
      ))}
    </ContextSubMenu>
  );
};

const ChangeBoardMenuItem = ({ board, onMove }: { board: GalleryBoard; onMove: (boardId: string) => void }) => {
  const handleClick = useCallback(() => onMove(board.id), [board.id, onMove]);

  return (
    <Menu.Item value={`move-to-${board.id}`} onClick={handleClick}>
      <Text fontSize="xs" minW="0" truncate>
        {board.name}
      </Text>
    </Menu.Item>
  );
};

const ContextSubMenu = ({
  children,
  icon,
  label,
  scrollArea,
}: {
  children: ReactNode;
  icon: LucideIcon;
  label: string;
  scrollArea?: boolean;
}) => (
  <Menu.Root positioning={MENU_POSITIONING_PROPS}>
    <Menu.TriggerItem>
      <HStack gap="2" minW="0" w="full">
        <Icon as={icon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
        <Text flex="1" fontSize="xs">
          {label}
        </Text>
        <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
      </HStack>
    </Menu.TriggerItem>
    <Portal>
      <Menu.Positioner>
        <MenuContent {...MENU_CONTENT_PROPS} maxH="18rem" overflowY={scrollArea ? undefined : 'auto'}>
          {scrollArea ? (
            <ScrollArea.Root maxH="inherit" size="xs" variant="hover" w="full">
              <ScrollArea.Viewport maxH="inherit" w="full">
                <ScrollArea.Content>{children}</ScrollArea.Content>
              </ScrollArea.Viewport>
              <ScrollArea.Scrollbar>
                <ScrollArea.Thumb />
              </ScrollArea.Scrollbar>
            </ScrollArea.Root>
          ) : (
            children
          )}
        </MenuContent>
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);

const QuickMenuItem = ({
  icon,
  label,
  value,
  onClick,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  onClick: () => void;
}) => (
  <Tooltip
    showArrow
    content={label}
    contentProps={QUICK_MENU_TOOLTIP_CONTENT_PROPS}
    openDelay={300}
    positioning={QUICK_MENU_TOOLTIP_POSITIONING_PROPS}
  >
    <Menu.Item aria-label={label} flex="1" justifyContent="center" value={value} onClick={onClick}>
      <Icon as={icon} boxSize="4" color="fg" />
    </Menu.Item>
  </Tooltip>
);

const ContextMenuItem = ({
  color,
  disabled,
  icon,
  label,
  value,
  onClick,
}: {
  color?: string;
  disabled?: boolean;
  icon: LucideIcon;
  label: string;
  value: string;
  onClick?: () => void;
}) => (
  <Menu.Item color={color} disabled={disabled} value={value} onClick={onClick}>
    <HStack gap="2" minW="0" w="full">
      <Icon as={icon} boxSize="3.5" color={color ?? 'fg.subtle'} flexShrink={0} />
      <Text flex="1" fontSize="xs">
        {label}
      </Text>
    </HStack>
  </Menu.Item>
);
