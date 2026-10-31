import type { GalleryBoard, GalleryImage, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type { GalleryItemContextMenuTarget } from '@features/gallery/react';
import type { ModelsSnapshot } from '@features/models';
import type { SpandrelModelConfig } from '@features/upscale';
import type { VideoConditioningRole } from '@features/video';
import type { GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';

import { HStack, Icon, Menu, Portal, ScrollArea, Text } from '@chakra-ui/react';
import {
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  toGalleryItemKey,
} from '@features/gallery';
import { galleryKeys } from '@features/gallery/queries';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { getModelOnlyUpscaleFactors, getSpandrelNativeScale } from '@features/upscale';
import { canCopyImage } from '@platform/browser/clipboard';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { MenuContent, MenuIconItem } from '@platform/ui';
import { MenuItem } from '@platform/ui/Menu';
import { useQuery } from '@tanstack/react-query';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectId, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import {
  ClapperboardIcon,
  AsteriskIcon,
  AudioLinesIcon,
  BoxIcon,
  ChevronRightIcon,
  CopyIcon,
  CrosshairIcon,
  DownloadIcon,
  ExternalLinkIcon,
  EyeIcon,
  FileImageIcon,
  FileJsonIcon,
  FilmIcon,
  FolderIcon,
  ImageIcon,
  ImagePlayIcon,
  ImagesIcon,
  LayersIcon,
  QuoteIcon,
  RulerIcon,
  ScanIcon,
  ScissorsIcon,
  ShuffleIcon,
  SlidersHorizontalIcon,
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

import { GalleryBoardDestinationGroups } from './GalleryBoardDestinationGroups';
import { EMPTY_IMAGE_RECALL_CAPABILITIES, type ImageRecallCapabilities } from './imageRecall';
import { EMPTY_VIDEO_RECALL_CAPABILITIES, type VideoRecallCapabilities, type VideoRecallKind } from './videoRecall';

/** What the menu learns about a video after it opens: metadata recall and whether a workflow is embedded. */
interface VideoItemCapabilities {
  recall: VideoRecallCapabilities;
  workflow: boolean;
}

const EMPTY_VIDEO_ITEM_CAPABILITIES: VideoItemCapabilities = {
  recall: EMPTY_VIDEO_RECALL_CAPABILITIES,
  workflow: false,
};

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

/** Shared image context menu anchored to a cursor rect; setting target opens it without a trigger element. */
export const ImageContextMenu = ({
  actions,
  boards,
  previewVideoActions,
  target,
  onClose,
}: {
  actions: ImageActions;
  /** The host's board listing as it shows it: folded through `useHostGalleryBoards` where boards cannot be shared. */
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

  // eslint-disable-next-line react/refs
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
                allStarred={target.allStarred}
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
  const owner = captureAccountScope();
  const videoItem = item.kind === 'video' ? item : null;
  // The record answers whether a workflow is embedded only when the listing did not say.
  const videoCapabilitiesQuery = useQuery({
    enabled: videoItem !== null,
    queryKey: galleryKeys.videoMenuCapabilities(owner, videoItem?.name ?? null),
    queryFn: async ({ signal }): Promise<VideoItemCapabilities> => {
      if (!videoItem) {
        return EMPTY_VIDEO_ITEM_CAPABILITIES;
      }

      const [recall, workflow] = await Promise.all([
        actions.getVideoRecallCapabilities(videoItem, signal),
        actions.hasVideoWorkflow(videoItem, signal),
      ]);

      return { recall, workflow };
    },
  });
  const isLoadingVideoRecall = videoCapabilitiesQuery.isPending;
  const { recall: videoRecallCapabilities, workflow: hasVideoWorkflow } =
    videoCapabilitiesQuery.data ?? EMPTY_VIDEO_ITEM_CAPABILITIES;
  const handleLoadVideoWorkflow = useCallback(() => {
    if (item.kind === 'video') {
      void actions.loadVideoWorkflow(item);
    }
  }, [actions, item]);

  const makeVideoRecallHandler = useCallback(
    (kind: VideoRecallKind) => () => {
      if (item.kind === 'video') {
        void actions.recallVideoData(item, kind);
      }
    },
    [actions, item]
  );
  const handleVideoRecallAll = useMemo(() => makeVideoRecallHandler('all'), [makeVideoRecallHandler]);
  const handleVideoRecallRemix = useMemo(() => makeVideoRecallHandler('remix'), [makeVideoRecallHandler]);
  const handleVideoRecallPrompts = useMemo(() => makeVideoRecallHandler('prompts'), [makeVideoRecallHandler]);
  const handleVideoRecallSeed = useMemo(() => makeVideoRecallHandler('seed'), [makeVideoRecallHandler]);
  const handleExtendInVideo = useCallback(() => {
    if (item.kind === 'video') {
      actions.sendToInitialVideo(item);
    }
  }, [actions, item]);
  const handleUseAsReferenceVideo = useCallback(() => {
    if (item.kind === 'video') {
      // eslint-disable-next-line react/hooks -- `useAsReferenceVideo` is a plain action method; the compiler reads its name as a hook.
      actions.useAsReferenceVideo(item);
    }
  }, [actions, item]);
  const makeConditioningClipHandler = useCallback(
    (role: VideoConditioningRole) => () => {
      if (item.kind === 'video') {
        // eslint-disable-next-line react/hooks -- `useAsConditioningClip` is a plain action method; the compiler reads its name as a hook.
        actions.useAsConditioningClip(item, role);
      }
    },
    [actions, item]
  );
  const handleUseSoundtrack = useMemo(() => makeConditioningClipHandler('audio'), [makeConditioningClipHandler]);
  const handleUsePicture = useMemo(() => makeConditioningClipHandler('video'), [makeConditioningClipHandler]);

  return (
    <>
      <HStack gap="1">
        <MenuIconItem
          icon={ExternalLinkIcon}
          label="Open in new tab"
          value="open-in-new-tab"
          onSelect={handleOpenInNewTab}
        />
        <MenuIconItem
          icon={DownloadIcon}
          label={`Download ${mediaLabel}`}
          value="download-item"
          onSelect={handleDownload}
        />
        <MenuIconItem icon={EyeIcon} label="Open in preview" value="open-in-preview" onSelect={handleOpenPreview} />
        <MenuIconItem
          icon={StarIcon}
          iconFill={item.starred ? 'currentColor' : 'none'}
          label={`${item.starred ? 'Unstar' : 'Star'} ${mediaLabel}`}
          value="toggle-starred"
          onSelect={handleToggleStarred}
        />
      </HStack>
      <Menu.Separator borderColor="border.subtle" />
      {item.kind === 'video' ? (
        <>
          <ContextMenuItem
            disabled={isLoadingVideoRecall || !hasVideoWorkflow}
            icon={WorkflowIcon}
            label={t('widgets.gallery.itemActions.loadWorkflow.label')}
            value="load-video-workflow"
            onClick={handleLoadVideoWorkflow}
          />
          <ContextSubMenu icon={AsteriskIcon} label="Recall Metadata">
            <ContextMenuItem
              disabled={isLoadingVideoRecall || !videoRecallCapabilities.all}
              icon={AsteriskIcon}
              label="Recall All"
              value="video-recall-all"
              onClick={handleVideoRecallAll}
            />
            <ContextMenuItem
              disabled={isLoadingVideoRecall || !videoRecallCapabilities.remix}
              icon={ShuffleIcon}
              label="Remix Video"
              value="video-remix"
              onClick={handleVideoRecallRemix}
            />
            <ContextMenuItem
              disabled={isLoadingVideoRecall || !videoRecallCapabilities.prompts}
              icon={QuoteIcon}
              label="Use Prompt"
              value="video-use-prompt"
              onClick={handleVideoRecallPrompts}
            />
            <ContextMenuItem
              disabled={isLoadingVideoRecall || !videoRecallCapabilities.seed}
              icon={SproutIcon}
              label="Use Seed"
              value="video-use-seed"
              onClick={handleVideoRecallSeed}
            />
          </ContextSubMenu>
          <ContextMenuItem
            icon={ClapperboardIcon}
            label="Extend in Video"
            value="extend-in-video"
            onClick={handleExtendInVideo}
          />
          <ContextMenuItem
            disabled={!actions.canUseAsReferenceVideo}
            icon={FilmIcon}
            label="Use as Reference Video"
            value="use-as-reference-video"
            onClick={handleUseAsReferenceVideo}
          />
          <ContextSubMenu icon={AudioLinesIcon} label="Use as Conditioning Clip">
            <ContextMenuItem
              disabled={!actions.canUseAsConditioningClip.audio}
              icon={AudioLinesIcon}
              label={t('widgets.video.conditioningRoleAudio')}
              value="use-as-conditioning-audio"
              onClick={handleUseSoundtrack}
            />
            {/* A wrapped audio upload's picture is a placeholder, so it has nothing to lend. */}
            <ContextMenuItem
              disabled={!actions.canUseAsConditioningClip.video || item.mediaOrigin === 'audio_upload'}
              icon={ImagePlayIcon}
              label={t('widgets.video.conditioningRoleVideo')}
              value="use-as-conditioning-video"
              onClick={handleUsePicture}
            />
          </ContextSubMenu>
        </>
      ) : null}
      {item.kind === 'video' && previewVideoActions && previewVideoActions.itemKey === toGalleryItemKey(item) ? (
        <>
          <ContextMenuItem
            // Without a clipboard (plain HTTP) the item says why; a frame still loading only disables it.
            disabled={!previewVideoActions.isCopyCurrentFrameAvailable}
            icon={CopyIcon}
            label={t('widgets.preview.copyCurrentFrame')}
            unavailableReason={canCopyImage() ? undefined : t('common.copyUnavailable')}
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
      <FindInGalleryMenuItem itemRef={itemRef} />
      <ChangeBoardSubMenu boards={boards} currentBoardId={item.boardId} onMove={handleMove} />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        isDanger
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
  allStarred: knownAllStarred,
  boards,
  itemRefs,
  loadedItems,
  primaryItem,
  onRequestDeletion,
}: {
  actions: ImageActions;
  allStarred: boolean | undefined;
  boards: GalleryBoard[];
  itemRefs: GalleryItemRef[];
  loadedItems: GalleryItem[];
  primaryItem: GalleryItem;
  onRequestDeletion: (itemRefs: GalleryItemRef[]) => void;
}) => {
  const allStarred =
    knownAllStarred ?? (loadedItems.length === itemRefs.length && loadedItems.every((item) => item.starred));
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
      <Text color="fg.subtle" fontSize="xs" fontWeight="700" px="3" py="1.5" textTransform="uppercase">
        {itemRefs.length} items selected
      </Text>
      <Menu.Separator borderColor="border.subtle" />
      <HStack gap="1">
        <MenuIconItem
          icon={ExternalLinkIcon}
          label="Open in new tab"
          value="open-primary-in-new-tab"
          onSelect={handleOpenInNewTab}
        />
        <MenuIconItem
          icon={EyeIcon}
          label="Open in preview"
          value="open-primary-in-preview"
          onSelect={handleOpenPreview}
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
        isDanger
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

  // eslint-disable-next-line react/refs
  targetRef.current = target;

  const isBulk = images.length > 1;
  const imageNames = useMemo(() => images.map((candidate) => candidate.imageName), [images]);
  const recallRequestKey = getImageContextMenuRecallRequestKey(image, isBulk);
  const getImageRecallCapabilities = actions.getImageRecallCapabilities;

  // eslint-disable-next-line react/refs
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
  const { t } = useTranslation();
  const handleMove = useCallback(
    (boardId: string) => void actions.moveImagesToBoard([image.imageName], boardId),
    [actions, image.imageName]
  );
  const handleDelete = useCallback(() => onRequestDeletion([image.imageName]), [image.imageName, onRequestDeletion]);
  const imageItemRef = useMemo<GalleryItemRef>(() => ({ kind: 'image', name: image.imageName }), [image.imageName]);
  const handleRecallAll = useRecallImageDataHandler(actions, image, 'all');
  const handleRecallRemix = useRecallImageDataHandler(actions, image, 'remix');
  const handleRecallPrompts = useRecallImageDataHandler(actions, image, 'prompts');
  const handleRecallSeed = useRecallImageDataHandler(actions, image, 'seed');
  const handleRecallDimensions = useRecallImageDataHandler(actions, image, 'dimensions');
  const handleRecallClipSkip = useRecallImageDataHandler(actions, image, 'clipSkip');
  const handleSelectForCompare = useSelectForCompareHandler(actions, image);
  const handleSavePromptAsTemplate = useCallback(() => void actions.savePromptAsTemplate(image), [actions, image]);
  const handleUseAsReferenceImage = useUseAsReferenceImageHandler(actions, image);
  const handleUseAsControl = useCallback(() => void actions.sendImageAsControl(image), [actions, image]);
  const handleLoadWorkflow = useCallback(() => void actions.loadImageWorkflow(image), [actions, image]);
  const handleGeneratePbrMaps = useCallback(() => void actions.generatePbrMaps(image), [actions, image]);
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
    // The first frame, an initial video, and Ref2VA references all claim the same
    // conditioning slot: sending a frame clears the rivals.
    widgets.patchValues('video', {
      firstFrameImage: { height: image.height, image_name: image.imageName, width: image.width },
      references: [],
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
      <ContextMenuItem
        disabled={isLoadingRecallCapabilities || !recallCapabilities.workflow}
        icon={WorkflowIcon}
        label={t('widgets.gallery.itemActions.loadWorkflow.label')}
        value="load-workflow"
        onClick={handleLoadWorkflow}
      />
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
      <CanvasDestinationMenuItems actions={actions} images={images} isBulk={false} />
      <UpscaleSubMenu image={image} onSendToUpscale={handleSendToUpscale} />
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
        icon={SlidersHorizontalIcon}
        label={t('widgets.gallery.itemActions.useAsControl')}
        value="use-as-control"
        onClick={handleUseAsControl}
      />
      <ContextMenuItem
        // Use the same availability signal as Use Prompt to disable empty recall upfront.
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
      <ContextMenuItem
        icon={BoxIcon}
        label={t('widgets.gallery.itemActions.pbrMaps.label')}
        value="generate-pbr-maps"
        onClick={handleGeneratePbrMaps}
      />
      <Menu.Separator borderColor="border.subtle" />
      <FindInGalleryMenuItem itemRef={imageItemRef} />
      <ChangeBoardSubMenu boards={boards} currentBoardId={image.boardId} onMove={handleMove} />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem isDanger icon={Trash2Icon} label="Delete Image" value="delete-image" onClick={handleDelete} />
    </>
  );
};

type RecallImageDataKind = Parameters<ImageActions['recallImageData']>[1];

const useRecallImageDataHandler = (actions: ImageActions, image: GalleryImage, kind: RecallImageDataKind) =>
  useCallback(() => void actions.recallImageData(image, kind), [actions, image, kind]);

const useSelectForCompareHandler = (actions: ImageActions, image: GalleryImage) =>
  useCallback(() => actions.selectForCompare(image), [actions, image]);

const useUseAsReferenceImageHandler = (actions: ImageActions, image: GalleryImage) =>
  // eslint-disable-next-line react/hooks -- `useAsReferenceImage` is a plain action method; the compiler reads its name as a hook.
  useCallback(() => actions.useAsReferenceImage(image), [actions, image]);

const OpenInNewTabQuickMenuItem = ({ image }: { image: GalleryImage }) => {
  const handleClick = useCallback(() => window.open(image.imageUrl, '_blank', 'noopener'), [image.imageUrl]);

  return (
    <MenuIconItem icon={ExternalLinkIcon} label="Open in new tab" value="open-in-new-tab" onSelect={handleClick} />
  );
};

const CopyQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => void actions.copyImage(image), [actions, image]);

  return (
    <MenuIconItem
      icon={CopyIcon}
      label="Copy to clipboard"
      unavailableReason={canCopyImage() ? undefined : t('common.copyUnavailable')}
      value="copy-to-clipboard"
      onSelect={handleClick}
    />
  );
};

const DownloadQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const handleClick = useCallback(() => void actions.downloadImage(image), [actions, image]);

  return <MenuIconItem icon={DownloadIcon} label="Download image" value="download-image" onSelect={handleClick} />;
};

const OpenPreviewQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const handleClick = useCallback(() => actions.openImageInPreview(image), [actions, image]);

  return <MenuIconItem icon={EyeIcon} label="Open in preview" value="open-in-preview" onSelect={handleClick} />;
};

const ToggleStarQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const handleClick = useCallback(
    () => void actions.setImagesStarred([image.imageName], !image.starred),
    [actions, image.imageName, image.starred]
  );

  return (
    <MenuIconItem
      icon={StarIcon}
      iconFill={image.starred ? 'currentColor' : 'none'}
      label={image.starred ? 'Unstar image' : 'Star image'}
      value="toggle-starred"
      onSelect={handleClick}
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
      <Text color="fg.subtle" fontSize="xs" fontWeight="700" px="3" py="1.5" textTransform="uppercase">
        {images.length} images selected
      </Text>
      <Menu.Separator borderColor="border.subtle" />
      <CanvasDestinationMenuItems actions={actions} images={images} isBulk />
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
        isDanger
        icon={Trash2Icon}
        label="Delete Selection"
        value="delete-selection"
        onClick={handleDelete}
      />
    </>
  );
};

/** The open project's canvas leads; starting a separate project from the images is the explicit alternative. */
const CanvasDestinationMenuItems = ({
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
  const handleNewProject = useCallback(() => void actions.createCanvasFromImages(images), [actions, images]);

  return (
    <>
      <ContextSubMenu icon={LayersIcon} label={t('widgets.canvas.import.addToCurrentCanvas')}>
        {items.map((item) => (
          <GalleryCanvasImportDestinationMenuItem
            key={item.destination}
            actions={actions}
            images={images}
            item={item}
          />
        ))}
      </ContextSubMenu>
      <ContextMenuItem
        icon={FileImageIcon}
        label={t('widgets.canvas.import.newProjectFromImage', { count: images.length })}
        value="new-project-from-image"
        onClick={handleNewProject}
      />
    </>
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

const FindInGalleryMenuItem = ({ itemRef }: { itemRef: GalleryItemRef }) => {
  const { t } = useTranslation();
  const findGalleryItem = useFindGalleryItem();
  const handleClick = useCallback(() => findGalleryItem(itemRef), [findGalleryItem, itemRef]);

  return (
    <ContextMenuItem
      icon={CrosshairIcon}
      label={t('widgets.gallery.findInGallery')}
      value="find-in-gallery"
      onClick={handleClick}
    />
  );
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
  const { t } = useTranslation();
  const handleRemoveFromBoard = useCallback(() => onMove('none'), [onMove]);
  const handleSelectBoard = useCallback((board: GalleryBoard) => onMove(board.id), [onMove]);
  const destinations = useMemo(() => boards.filter((board) => board.id !== currentBoardId), [boards, currentBoardId]);

  return (
    <ContextSubMenu icon={FolderIcon} label={t('widgets.gallery.changeBoard')} scrollArea>
      {currentBoardId !== 'none' && (
        <ContextMenuItem
          icon={FolderIcon}
          label={t('widgets.gallery.removeFromBoard')}
          value="remove-from-board"
          onClick={handleRemoveFromBoard}
        />
      )}
      <GalleryBoardDestinationGroups boards={destinations} onSelect={handleSelectBoard} />
    </ContextSubMenu>
  );
};

/** Every installed upscaler, by name; a shallow-equal list keeps the menu from rerendering on unrelated model edits. */
const selectInstalledUpscalers = (snapshot: ModelsSnapshot): SpandrelModelConfig[] =>
  snapshot.models
    .filter((model) => model.type === 'spandrel_image_to_image')
    .map((model) => model as SpandrelModelConfig)
    .sort((a, b) => a.name.localeCompare(b.name));

const selectModelsStatus = (snapshot: ModelsSnapshot): ModelsSnapshot['status'] => snapshot.status;

/**
 * Send to Upscale opens the widget on the image; the items below it upscale the image at once with one upscaler alone,
 * at a factor picked from its submenu, into the gallery, leaving the widget as it is.
 */
const UpscaleSubMenu = ({ image, onSendToUpscale }: { image: GalleryImage; onSendToUpscale: () => void }) => {
  const { t } = useTranslation();
  const upscalers = useModelsSelector(selectInstalledUpscalers);
  const modelsStatus = useModelsSelector(selectModelsStatus);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  return (
    <ContextSubMenu icon={ScanIcon} label={t('widgets.gallery.itemActions.upscale.label')}>
      <ContextMenuItem
        icon={ScanIcon}
        label={t('widgets.gallery.itemActions.upscale.sendTo')}
        value="send-to-upscale"
        onClick={onSendToUpscale}
      />
      {upscalers.length === 0 ? (
        // "None installed" only once the library has loaded; until then the list is merely not known yet.
        <MenuItem
          unavailableReason={
            modelsStatus === 'loaded'
              ? t('widgets.gallery.itemActions.upscale.noUpscalers')
              : modelsStatus === 'error'
                ? t('models.couldNotLoad')
                : t('models.loadingModels')
          }
          value="upscale-model-only"
        >
          <Text fontSize="md">{t('widgets.gallery.itemActions.upscale.modelOnlyGroup')}</Text>
        </MenuItem>
      ) : (
        <>
          <Menu.Separator borderColor="border.subtle" />
          <Menu.ItemGroup>
            <Menu.ItemGroupLabel>{t('widgets.gallery.itemActions.upscale.modelOnlyGroup')}</Menu.ItemGroupLabel>
            {upscalers.map((upscaler) => (
              <ModelOnlyUpscalerSubMenu key={upscaler.key} image={image} upscaler={upscaler} />
            ))}
          </Menu.ItemGroup>
        </>
      )}
    </ContextSubMenu>
  );
};

/**
 * One upscaler's one-click targets: 1.5×, 2× and its native scale, as far as its plan reaches them. An upscaler whose
 * native scale was never recorded can still run once at it, but no other factor can be planned, so those say how to
 * record it.
 */
const ModelOnlyUpscalerSubMenu = ({ image, upscaler }: { image: GalleryImage; upscaler: SpandrelModelConfig }) => {
  const { t } = useTranslation();
  const factors = getModelOnlyUpscaleFactors(upscaler);
  const nativeScale = getSpandrelNativeScale(upscaler);

  return (
    <ContextSubMenu
      detail={
        nativeScale === null ? undefined : t('widgets.gallery.itemActions.upscale.scaleSuffix', { scale: nativeScale })
      }
      label={upscaler.name}
    >
      {factors ? (
        factors.map((factor) => (
          <ModelOnlyUpscaleFactorItem
            key={factor}
            image={image}
            label={t(
              factor === nativeScale
                ? 'widgets.gallery.itemActions.upscale.factorNative'
                : 'widgets.gallery.itemActions.upscale.scaleSuffix',
              { scale: factor }
            )}
            scale={factor === nativeScale ? 'native' : factor}
            upscaler={upscaler}
          />
        ))
      ) : (
        <>
          <ModelOnlyUpscaleFactorItem
            image={image}
            label={t('widgets.gallery.itemActions.upscale.nativeScaleUnknownFactor')}
            scale="native"
            upscaler={upscaler}
          />
          <MenuItem
            unavailableReason={t('widgets.upscale.nativeScaleUnknown')}
            value={`upscale-model-only-${upscaler.key}-other`}
          >
            <Text fontSize="md">{t('widgets.gallery.itemActions.upscale.otherFactors')}</Text>
          </MenuItem>
        </>
      )}
    </ContextSubMenu>
  );
};

const ModelOnlyUpscaleFactorItem = ({
  image,
  label,
  scale,
  upscaler,
}: {
  image: GalleryImage;
  label: string;
  /** `'native'` runs the upscaler once, whether or not its native scale is recorded. */
  scale: number | 'native';
  upscaler: SpandrelModelConfig;
}) => {
  const { t } = useTranslation();
  const commands = useWorkbenchCommands();
  const projectId = useActiveProjectId();
  const handleClick = useCallback(() => {
    // The account the pick was made under, before the command loads.
    const owner = captureAccountScope();

    // Loaded on pick, as Invoke's upscale submission loads it: the menu is part of the editor's startup graph.
    void import('./modelOnlyUpscale')
      .then(({ queueModelOnlyUpscale }) =>
        queueModelOnlyUpscale({
          commands,
          image: { height: image.height, image_name: image.imageName, width: image.width },
          owner,
          projectId,
          scale,
          t,
          upscaleModel: upscaler,
        })
      )
      // The command reports its own failures, so this is its chunk failing to load (a deploy replaced it, the
      // connection dropped): the pick must not go unanswered.
      .catch(() => {
        if (isAccountScopeCurrent(owner)) {
          commands.notifications.add({ kind: 'error', title: t('widgets.gallery.itemActions.upscale.loadFailed') });
        }
      });
  }, [commands, image.height, image.imageName, image.width, projectId, scale, t, upscaler]);

  return (
    <Menu.Item value={`upscale-model-only-${upscaler.key}-${scale}`} onClick={handleClick}>
      <Text fontSize="md" fontVariantNumeric="tabular-nums">
        {label}
      </Text>
    </Menu.Item>
  );
};

const ContextSubMenu = ({
  children,
  detail,
  icon,
  label,
  scrollArea,
}: {
  children: ReactNode;
  /** A short value after the label, such as an upscaler's native scale. */
  detail?: string;
  icon?: LucideIcon;
  label: string;
  scrollArea?: boolean;
}) => (
  <Menu.Root positioning={MENU_POSITIONING_PROPS}>
    <Menu.TriggerItem>
      <HStack gap="2" minW="0" w="full">
        {icon ? <Icon as={icon} boxSize="3.5" color="fg.subtle" flexShrink={0} /> : null}
        <Text flex="1" fontSize="md" minW="0" truncate>
          {label}
        </Text>
        {detail ? (
          // Secondary to the label, but at the menu's secondary-text contrast: fg.subtle is too faint on dark menus.
          <Text color="fg.muted" flexShrink={0} fontFamily="mono" fontSize="sm">
            {detail}
          </Text>
        ) : null}
        <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
      </HStack>
    </Menu.TriggerItem>
    <Portal>
      <Menu.Positioner>
        <MenuContent {...MENU_CONTENT_PROPS} maxH="18rem" overflowY={scrollArea ? undefined : 'auto'}>
          {scrollArea ? (
            <ScrollArea.Root maxH="inherit" variant="hover" w="full">
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

const ContextMenuItem = ({
  disabled,
  icon,
  iconFill,
  isDanger,
  label,
  unavailableReason,
  value,
  onClick,
}: {
  disabled?: boolean;
  icon: LucideIcon;
  /** Lucide icons are stroke-only, so `'currentColor'` is how an on state reads. */
  iconFill?: string;
  isDanger?: boolean;
  label: string;
  /** Why the item cannot act now; see `MenuItem`. */
  unavailableReason?: string;
  value: string;
  onClick?: () => void;
}) => (
  <MenuItem
    data-danger={isDanger ? '' : undefined}
    disabled={disabled}
    unavailableReason={unavailableReason}
    value={value}
    onClick={onClick}
  >
    <HStack gap="2" minW="0" w="full">
      <Icon as={icon} boxSize="3.5" color={isDanger ? undefined : 'fg.subtle'} fill={iconFill} flexShrink={0} />
      <Text flex="1" fontSize="md">
        {label}
      </Text>
    </HStack>
  </MenuItem>
);
