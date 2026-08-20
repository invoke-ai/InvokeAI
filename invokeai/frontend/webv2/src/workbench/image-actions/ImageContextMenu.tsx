import type { GalleryBoard, GalleryImage, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type { GalleryItemContextMenuTarget } from '@features/gallery/react';
/* eslint-disable react/react-compiler */
import type { GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';

import { HStack, Icon, Menu, Portal, ScrollArea, Text } from '@chakra-ui/react';
import {
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  toGalleryItemKey,
} from '@features/gallery';
import { createVideoSourceClip } from '@features/video';
import { MenuContent, Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import {
  ClapperboardIcon,
  AsteriskIcon,
  ChevronRightIcon,
  CopyIcon,
  DownloadIcon,
  ExternalLinkIcon,
  EyeIcon,
  FileImageIcon,
  FileJsonIcon,
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

export interface LegacyImageContextMenuTarget {
  /** Right-clicked image first; more entries switch the menu into bulk mode. */
  images: GalleryImage[];
  x: number;
  y: number;
}

export type ImageContextMenuTarget = LegacyImageContextMenuTarget | GalleryItemContextMenuTarget;

/**
 * Extra actions owned by Preview's selected native video. Other hosts omit
 * this port, so gallery/grid video menus retain their common-action-only
 * contract and never gain a frame-copy item.
 */
export interface PreviewVideoContextActions {
  isCopyCurrentFrameAvailable: boolean;
  itemKey: GalleryItemKey;
  onCopyCurrentFrame: () => void;
  onOpenDetails: () => void;
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
  if (target && 'itemRefs' in target) {
    const loadedItemsByKey = new Map(target.items.map((item) => [toGalleryItemKey(item), item]));
    const imageItems = target.itemRefs.flatMap((ref) => {
      const item = loadedItemsByKey.get(toGalleryItemKey(ref));

      return item && isGalleryImageItem(item) ? [item] : [];
    });

    return imageItems.length === target.itemRefs.length ? imageItems.map(galleryImageItemToGalleryImage) : [];
  }

  if (!Array.isArray(target?.images)) {
    return [];
  }

  return target.images.filter(isUsableGalleryImage);
};

const toGalleryItemContextMenuTarget = (target: ImageContextMenuTarget | null): GalleryItemContextMenuTarget | null => {
  if (!target) {
    return null;
  }

  if ('itemRefs' in target) {
    return target;
  }

  const images = getImageContextMenuImages(target);
  const items = images.map(legacyGeneratedImageToGalleryItem);

  return {
    itemRefs: items.map(({ kind, name }) => ({ kind, name })),
    items,
    x: target.x,
    y: target.y,
  };
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
  previewVideoActions,
  target,
  onClose,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  previewVideoActions?: PreviewVideoContextActions;
  target: ImageContextMenuTarget | null;
  onClose: () => void;
}) => {
  const itemTarget = toGalleryItemContextMenuTarget(target);
  const images = getImageContextMenuImages(itemTarget);
  const isCompleteImageTarget = Boolean(itemTarget && images.length === itemTarget.itemRefs.length);
  const imageTarget = useMemo<LegacyImageContextMenuTarget | null>(
    () => (itemTarget && isCompleteImageTarget ? { images, x: itemTarget.x, y: itemTarget.y } : null),
    [images, isCompleteImageTarget, itemTarget]
  );
  const requestDeletion = useCallback((itemRefs: GalleryItemRef[]) => void actions.deleteItems(itemRefs), [actions]);
  const requestImageDeletion = useCallback(
    (imageNames: string[]) => requestDeletion(imageNames.map((name) => ({ kind: 'image', name }))),
    [requestDeletion]
  );

  if (!itemTarget) {
    return null;
  }

  return imageTarget ? (
    <ImageContextMenuContent
      key={`${images[0]?.imageName ?? 'image'}:${images.length}`}
      actions={actions}
      boards={boards}
      image={images[0] ?? null}
      images={images}
      target={imageTarget}
      onClose={onClose}
      onRequestDeletion={requestImageDeletion}
    />
  ) : (
    <GalleryItemContextMenuContent
      key={`${itemTarget.itemRefs[0] ? toGalleryItemKey(itemTarget.itemRefs[0]) : 'item'}:${itemTarget.itemRefs.length}`}
      actions={actions}
      boards={boards}
      previewVideoActions={previewVideoActions}
      target={itemTarget}
      onClose={onClose}
      onRequestDeletion={requestDeletion}
    />
  );
};

const GalleryItemContextMenuContent = ({
  actions,
  boards,
  previewVideoActions,
  target,
  onClose,
  onRequestDeletion,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  previewVideoActions?: PreviewVideoContextActions;
  target: GalleryItemContextMenuTarget;
  onClose: () => void;
  onRequestDeletion: (itemRefs: GalleryItemRef[]) => void;
}) => {
  const targetRef = useRef(target);
  const item = target.items[0] ?? null;
  const itemRef = target.itemRefs[0] ?? null;
  const isBulk = target.itemRefs.length > 1;

  targetRef.current = target;

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

  if (!item || !itemRef) {
    return null;
  }

  return (
    <Menu.Root lazyMount open positioning={positioning} unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Menu.Positioner>
          <MenuContent {...MENU_CONTENT_PROPS} maxH="min(28rem, calc(100vh - 2rem))" minW="16rem" overflowY="auto">
            {isBulk ? (
              <BulkItemMenuItems
                actions={actions}
                boards={boards}
                itemRefs={target.itemRefs}
                loadedItems={target.items}
                primaryItem={item}
                onRequestDeletion={onRequestDeletion}
              />
            ) : (
              <SingleItemMenuItems
                actions={actions}
                boards={boards}
                item={item}
                itemRef={itemRef}
                previewVideoActions={previewVideoActions}
                onRequestDeletion={onRequestDeletion}
              />
            )}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const SingleItemMenuItems = ({
  actions,
  boards,
  item,
  itemRef,
  previewVideoActions,
  onRequestDeletion,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  item: GalleryItem;
  itemRef: GalleryItemRef;
  previewVideoActions?: PreviewVideoContextActions;
  onRequestDeletion: (itemRefs: GalleryItemRef[]) => void;
}) => {
  const { t } = useTranslation();
  const mediaLabel = item.kind === 'video' ? 'video' : 'item';
  const handleOpenInNewTab = useCallback(() => actions.openItemInNewTab(item), [actions, item]);
  const handleDownload = useCallback(() => void actions.downloadItem(item), [actions, item]);
  const handleOpenPreview = useCallback(() => actions.openItemInPreview(item), [actions, item]);
  const handleToggleStarred = useCallback(
    () => void actions.setItemsStarred([itemRef], !item.starred),
    [actions, item.starred, itemRef]
  );
  const handleMove = useCallback(
    (boardId: string) => void actions.moveItemsToBoard([itemRef], boardId),
    [actions, itemRef]
  );
  const handleDelete = useCallback(() => onRequestDeletion([itemRef]), [itemRef, onRequestDeletion]);
  const { generation, widgets } = useWorkbenchCommands();
  const openWidget = useOpenWorkbenchWidget();
  const handleExtendInVideo = useCallback(() => {
    if (item.kind !== 'video') {
      return;
    }

    openWidget('video', { preferredRegions: ['left'] });
    // An initial video and a first frame are mutually exclusive in the panel.
    widgets.patchValues('video', {
      firstFrameImage: null,
      sourceVideo: createVideoSourceClip({
        durationSeconds: item.durationSeconds,
        fps: item.fps,
        height: item.height,
        name: item.name,
        width: item.width,
      }),
    });
    generation.setSource('video');
  }, [generation, item, openWidget, widgets]);

  return (
    <>
      <HStack gap="1">
        <QuickMenuItem
          icon={ExternalLinkIcon}
          label="Open in new tab"
          value="open-in-new-tab"
          onClick={handleOpenInNewTab}
        />
        <QuickMenuItem
          icon={DownloadIcon}
          label={`Download ${mediaLabel}`}
          value="download-item"
          onClick={handleDownload}
        />
        <QuickMenuItem icon={EyeIcon} label="Open in preview" value="open-in-preview" onClick={handleOpenPreview} />
        <QuickMenuItem
          icon={StarIcon}
          iconFill={item.starred ? 'currentColor' : 'none'}
          label={`${item.starred ? 'Unstar' : 'Star'} ${mediaLabel}`}
          value="toggle-starred"
          onClick={handleToggleStarred}
        />
      </HStack>
      <Menu.Separator borderColor="border.subtle" />
      {item.kind === 'video' ? (
        <ContextMenuItem
          icon={ClapperboardIcon}
          label="Extend in Video"
          value="extend-in-video"
          onClick={handleExtendInVideo}
        />
      ) : null}
      {item.kind === 'video' && previewVideoActions && previewVideoActions.itemKey === toGalleryItemKey(item) ? (
        <>
          <ContextMenuItem
            disabled={!previewVideoActions.isCopyCurrentFrameAvailable}
            icon={CopyIcon}
            label={t('widgets.preview.copyCurrentFrame')}
            value="copy-current-video-frame"
            onClick={previewVideoActions.onCopyCurrentFrame}
          />
          <ContextMenuItem
            icon={FileJsonIcon}
            label={t('widgets.preview.videoDetails')}
            value="open-video-details"
            onClick={previewVideoActions.onOpenDetails}
          />
          <Menu.Separator borderColor="border.subtle" />
        </>
      ) : null}
      <ChangeBoardSubMenu boards={boards} currentBoardId={item.boardId} onMove={handleMove} />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        color="fg.error"
        icon={Trash2Icon}
        label={item.kind === 'video' ? 'Delete Video' : 'Delete Item'}
        value="delete-item"
        onClick={handleDelete}
      />
    </>
  );
};

const BulkItemMenuItems = ({
  actions,
  boards,
  itemRefs,
  loadedItems,
  primaryItem,
  onRequestDeletion,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  itemRefs: GalleryItemRef[];
  loadedItems: GalleryItem[];
  primaryItem: GalleryItem;
  onRequestDeletion: (itemRefs: GalleryItemRef[]) => void;
}) => {
  const allStarred = loadedItems.length === itemRefs.length && loadedItems.every((item) => item.starred);
  const handleOpenInNewTab = useCallback(() => actions.openItemInNewTab(primaryItem), [actions, primaryItem]);
  const handleOpenPreview = useCallback(() => actions.openItemInPreview(primaryItem), [actions, primaryItem]);
  const handleToggleStarred = useCallback(
    () => void actions.setItemsStarred(itemRefs, !allStarred),
    [actions, allStarred, itemRefs]
  );
  const handleDownload = useCallback(
    () => void actions.downloadItems(itemRefs, loadedItems),
    [actions, itemRefs, loadedItems]
  );
  const handleMove = useCallback(
    (boardId: string) => void actions.moveItemsToBoard(itemRefs, boardId),
    [actions, itemRefs]
  );
  const handleDelete = useCallback(() => onRequestDeletion(itemRefs), [itemRefs, onRequestDeletion]);

  return (
    <>
      <Text color="fg.subtle" fontSize="2xs" fontWeight="700" px="3" py="1.5" textTransform="uppercase">
        {itemRefs.length} items selected
      </Text>
      <Menu.Separator borderColor="border.subtle" />
      <HStack gap="1">
        <QuickMenuItem
          icon={ExternalLinkIcon}
          label="Open in new tab"
          value="open-primary-in-new-tab"
          onClick={handleOpenInNewTab}
        />
        <QuickMenuItem
          icon={EyeIcon}
          label="Open in preview"
          value="open-primary-in-preview"
          onClick={handleOpenPreview}
        />
      </HStack>
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        icon={StarIcon}
        iconFill={allStarred ? 'currentColor' : 'none'}
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

const ImageContextMenuContent = ({
  actions,
  boards,
  image,
  images,
  target,
  onClose,
  onRequestDeletion,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  image: GalleryImage;
  images: GalleryImage[];
  target: ImageContextMenuTarget;
  onClose: () => void;
  onRequestDeletion: (imageNames: string[]) => void;
}) => {
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
  return (
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
                onRequestDeletion={onRequestDeletion}
              />
            ) : (
              <SingleImageMenuItems
                actions={actions}
                boards={boards}
                image={image}
                images={images}
                isLoadingRecallCapabilities={isLoadingRecallCapabilities}
                onRequestDeletion={onRequestDeletion}
                recallCapabilities={recallCapabilities}
              />
            )}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
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
  const handleSavePromptAsTemplate = useCallback(() => void actions.savePromptAsTemplate(image), [actions, image]);
  const handleUseAsReferenceImage = useUseAsReferenceImageHandler(actions, image);
  const { generation, widgets } = useWorkbenchCommands();
  const openWidget = useOpenWorkbenchWidget();
  const handleSendToUpscale = useCallback(() => {
    openWidget('upscale', { preferredRegions: ['left'] });
    widgets.patchValues('upscale', {
      inputImage: { height: image.height, image_name: image.imageName, width: image.width },
    });
    generation.setSource('upscale');
  }, [generation, image.height, image.imageName, image.width, openWidget, widgets]);
  const handleSendToVideo = useCallback(() => {
    openWidget('video', { preferredRegions: ['left'] });
    // The first frame and an initial video are mutually exclusive in the panel.
    widgets.patchValues('video', {
      firstFrameImage: { height: image.height, image_name: image.imageName, width: image.width },
      sourceVideo: null,
    });
    generation.setSource('video');
  }, [generation, image.height, image.imageName, image.width, openWidget, widgets]);

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
        icon={ClapperboardIcon}
        label="Send to Video"
        value="send-to-video"
        onClick={handleSendToVideo}
      />
      <ContextMenuItem
        disabled={!actions.canUseAsReferenceImage}
        icon={ImageIcon}
        label="Use as Reference Image"
        value="use-as-reference-image"
        onClick={handleUseAsReferenceImage}
      />
      <ContextMenuItem
        // Same signal "Use Prompt" reads, rather than staying enabled and
        // raising a toast to say the image had no prompt after all.
        disabled={isLoadingRecallCapabilities || !recallCapabilities.prompts}
        icon={TypeIcon}
        label="Use as Prompt Template"
        value="use-as-prompt-template"
        onClick={handleSavePromptAsTemplate}
      />
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
      iconFill={image.starred ? 'currentColor' : 'none'}
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
        iconFill={allStarred ? 'currentColor' : 'none'}
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
      <MiddleTruncate fontSize="xs" minW="0" text={board.name} />
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
  iconFill,
  label,
  value,
  onClick,
}: {
  icon: LucideIcon;
  /** Lucide icons are stroke-only, so `'currentColor'` is how an on state reads. */
  iconFill?: string;
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
      <Icon as={icon} boxSize="4" color="fg" fill={iconFill} />
    </Menu.Item>
  </Tooltip>
);

const ContextMenuItem = ({
  color,
  disabled,
  icon,
  iconFill,
  label,
  value,
  onClick,
}: {
  color?: string;
  disabled?: boolean;
  icon: LucideIcon;
  /** Lucide icons are stroke-only, so `'currentColor'` is how an on state reads. */
  iconFill?: string;
  label: string;
  value: string;
  onClick?: () => void;
}) => (
  <Menu.Item color={color} disabled={disabled} value={value} onClick={onClick}>
    <HStack gap="2" minW="0" w="full">
      <Icon as={icon} boxSize="3.5" color={color ?? 'fg.subtle'} fill={iconFill} flexShrink={0} />
      <Text flex="1" fontSize="xs">
        {label}
      </Text>
    </HStack>
  </Menu.Item>
);
