import type { GalleryBoard, GalleryImage, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type { GalleryItemContextMenuTarget } from '@features/gallery/react';
import type { VideoConditioningRole } from '@features/video';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import {
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  toGalleryItemKey,
} from '@features/gallery';
import { getGalleryMoveTargets } from '@features/gallery/contracts';
import { galleryKeys } from '@features/gallery/queries';
import { canCopyImage } from '@platform/browser/clipboard';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { MenuContent, MenuIconItem } from '@platform/ui';
import { MenuItem } from '@platform/ui/Menu';
import { useQuery } from '@tanstack/react-query';
import { formatHotkeyPartLabel } from '@workbench/hotkeys/keys';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import {
  ClapperboardIcon,
  AsteriskIcon,
  AudioLinesIcon,
  BoxIcon,
  CopyIcon,
  CrosshairIcon,
  DownloadIcon,
  ExternalLinkIcon,
  EyeIcon,
  FileJsonIcon,
  FilmIcon,
  FolderIcon,
  ImagePlayIcon,
  ImagesIcon,
  QuoteIcon,
  ShuffleIcon,
  SproutIcon,
  StarIcon,
  Trash2Icon,
  WorkflowIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { ImageRecallCapabilities } from './imageRecall';
import type { ImageActions } from './useImageActions';

import { ContextSubMenu, IMAGE_MENU_CONTENT_PROPS } from './ContextSubMenu';
import { GalleryBoardDestinationGroups } from './GalleryBoardDestinationGroups';
import { ImageContinueGroup, ImageUseGroups } from './ImageItemMenuGroups';
import { useImageRecallCapabilities } from './useImageRecallCapabilities';
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
 * `housekeeping` leaves out the groups a host already offers beside the menu (Preview's Edit and Use menus): what to
 * continue working on with the image and what to take from it. Single images only; videos and selections stay full.
 */
export type ImageContextMenuVariant = 'full' | 'housekeeping';

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

const isUsableGalleryImage = (value: unknown): value is GalleryImage =>
  Boolean(value) &&
  typeof value === 'object' &&
  typeof (value as GalleryImage).imageName === 'string' &&
  typeof (value as GalleryImage).imageUrl === 'string' &&
  typeof (value as GalleryImage).thumbnailUrl === 'string';

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

/** Shared image context menu anchored to a cursor rect; setting target opens it without a trigger element. */
export const ImageContextMenu = ({
  actions,
  boards,
  previewVideoActions,
  target,
  variant = 'full',
  onClose,
}: {
  actions: ImageActions;
  /** The host's board listing as it shows it: folded through `useHostGalleryBoards` where boards cannot be shared. */
  boards: GalleryBoard[];
  previewVideoActions?: PreviewVideoContextActions;
  target: ImageContextMenuTarget | null;
  variant?: ImageContextMenuVariant;
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
      variant={variant}
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
          <MenuContent
            {...IMAGE_MENU_CONTENT_PROPS}
            maxH="min(28rem, calc(100vh - 2rem))"
            minW="16rem"
            overflowY="auto"
          >
            {isBulk ? (
              <BulkItemMenuItems
                actions={actions}
                allStarred={target.allStarred}
                boards={boards}
                itemRefs={target.itemRefs}
                listingBoardId={target.listingBoardId}
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
  const isVideo = item.kind === 'video';
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
  const singleSourceBoardIds = useMemo(() => [item.boardId], [item.boardId]);
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
          label={t('widgets.gallery.itemMenu.openInNewTab')}
          value="open-in-new-tab"
          onSelect={handleOpenInNewTab}
        />
        <MenuIconItem
          icon={DownloadIcon}
          label={t(isVideo ? 'widgets.gallery.itemMenu.downloadVideo' : 'widgets.gallery.itemMenu.downloadImage')}
          value="download-item"
          onSelect={handleDownload}
        />
        <MenuIconItem
          icon={EyeIcon}
          label={t('widgets.gallery.itemMenu.openInPreview')}
          value="open-in-preview"
          onSelect={handleOpenPreview}
        />
        <MenuIconItem
          icon={StarIcon}
          iconFill={item.starred ? 'currentColor' : 'none'}
          label={t(
            isVideo
              ? item.starred
                ? 'widgets.preview.unstarVideo'
                : 'widgets.preview.starVideo'
              : item.starred
                ? 'widgets.preview.unstarImage'
                : 'widgets.preview.starImage'
          )}
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
          <ContextSubMenu icon={AsteriskIcon} label={t('widgets.gallery.itemMenu.reuseFromVideo')}>
            <ContextMenuItem
              disabled={isLoadingVideoRecall || !videoRecallCapabilities.all}
              icon={AsteriskIcon}
              label={t('widgets.gallery.itemMenu.allSettings')}
              value="video-recall-all"
              onClick={handleVideoRecallAll}
            />
            <ContextMenuItem
              disabled={isLoadingVideoRecall || !videoRecallCapabilities.remix}
              icon={ShuffleIcon}
              label={t('widgets.gallery.itemMenu.remixVideo')}
              value="video-remix"
              onClick={handleVideoRecallRemix}
            />
            <ContextMenuItem
              disabled={isLoadingVideoRecall || !videoRecallCapabilities.prompts}
              icon={QuoteIcon}
              label={t('widgets.gallery.itemActions.recall.verbs.prompts')}
              value="video-use-prompt"
              onClick={handleVideoRecallPrompts}
            />
            <ContextMenuItem
              disabled={isLoadingVideoRecall || !videoRecallCapabilities.seed}
              icon={SproutIcon}
              label={t('widgets.gallery.itemActions.recall.verbs.seed')}
              value="video-use-seed"
              onClick={handleVideoRecallSeed}
            />
          </ContextSubMenu>
          <ContextMenuItem
            icon={ClapperboardIcon}
            label={t('widgets.gallery.itemMenu.extendInVideo')}
            value="extend-in-video"
            onClick={handleExtendInVideo}
          />
          <ContextMenuItem
            disabled={!actions.canUseAsReferenceVideo}
            icon={FilmIcon}
            label={t('widgets.gallery.itemMenu.useAsReferenceVideo')}
            value="use-as-reference-video"
            onClick={handleUseAsReferenceVideo}
          />
          <ContextSubMenu icon={AudioLinesIcon} label={t('widgets.gallery.itemMenu.useAsConditioningClip')}>
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
      <ChangeBoardSubMenu boards={boards} itemBoardIds={singleSourceBoardIds} onMove={handleMove} />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        isDanger
        icon={Trash2Icon}
        label={t(isVideo ? 'widgets.gallery.itemMenu.deleteVideo' : 'widgets.gallery.itemMenu.deleteImage')}
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
  listingBoardId,
  loadedItems,
  primaryItem,
  onRequestDeletion,
}: {
  actions: ImageActions;
  allStarred: boolean | undefined;
  boards: GalleryBoard[];
  itemRefs: GalleryItemRef[];
  /** The board the listing the selection was made in reads, where items no page holds sit. */
  listingBoardId: string | undefined;
  loadedItems: GalleryItem[];
  primaryItem: GalleryItem;
  onRequestDeletion: (itemRefs: GalleryItemRef[]) => void;
}) => {
  const { t } = useTranslation();
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
  const itemBoardIds = useMemo(() => loadedItems.map((item) => item.boardId), [loadedItems]);

  return (
    <>
      <Text color="fg.subtle" fontSize="xs" fontWeight="700" px="3" py="1.5" textTransform="uppercase">
        {t('widgets.gallery.itemMenu.itemsSelected', { count: itemRefs.length })}
      </Text>
      <Menu.Separator borderColor="border.subtle" />
      <HStack gap="1">
        <MenuIconItem
          icon={ExternalLinkIcon}
          label={t('widgets.gallery.itemMenu.openInNewTab')}
          value="open-primary-in-new-tab"
          onSelect={handleOpenInNewTab}
        />
        <MenuIconItem
          icon={EyeIcon}
          label={t('widgets.gallery.itemMenu.openInPreview')}
          value="open-primary-in-preview"
          onSelect={handleOpenPreview}
        />
      </HStack>
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        icon={StarIcon}
        iconFill={allStarred ? 'currentColor' : 'none'}
        label={t(allStarred ? 'widgets.gallery.unstarSelection' : 'widgets.gallery.starSelection')}
        value="toggle-starred-all"
        onClick={handleToggleStarred}
      />
      <ContextMenuItem
        icon={DownloadIcon}
        label={t('widgets.gallery.downloadSelection')}
        value="download-selection"
        onClick={handleDownload}
      />
      <ChangeBoardSubMenu
        boards={boards}
        itemBoardIds={itemBoardIds}
        listingBoardId={listingBoardId}
        unloadedCount={itemRefs.length - loadedItems.length}
        onMove={handleMove}
      />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        isDanger
        icon={Trash2Icon}
        label={t('widgets.gallery.deleteSelection')}
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
  variant,
  onClose,
  onRequestDeletion,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  image: GalleryImage;
  images: GalleryImage[];
  target: ImageContextMenuTarget;
  variant: ImageContextMenuVariant;
  onClose: () => void;
  onRequestDeletion: (imageNames: string[]) => void;
}) => {
  const targetRef = useRef(target);

  // eslint-disable-next-line react/refs
  targetRef.current = target;

  const isBulk = images.length > 1;
  const imageNames = useMemo(() => images.map((candidate) => candidate.imageName), [images]);
  const { capabilities: recallCapabilities, isLoading: isLoadingRecallCapabilities } = useImageRecallCapabilities(
    actions,
    isBulk ? null : image
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
  return (
    <Menu.Root lazyMount open positioning={positioning} unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Menu.Positioner>
          <MenuContent
            {...IMAGE_MENU_CONTENT_PROPS}
            maxH="min(36rem, calc(100vh - 2rem))"
            minW="16rem"
            overflowY="auto"
          >
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
                recallCapabilities={recallCapabilities}
                variant={variant}
                onRequestDeletion={onRequestDeletion}
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
  recallCapabilities,
  variant,
  onRequestDeletion,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  image: GalleryImage;
  images: GalleryImage[];
  isLoadingRecallCapabilities: boolean;
  recallCapabilities: ImageRecallCapabilities;
  variant: ImageContextMenuVariant;
  onRequestDeletion: (imageNames: string[]) => void;
}) => {
  const { t } = useTranslation();
  const handleMove = useCallback(
    (boardId: string) => void actions.moveImagesToBoard([image.imageName], boardId),
    [actions, image.imageName]
  );
  const handleDelete = useCallback(() => onRequestDeletion([image.imageName]), [image.imageName, onRequestDeletion]);
  const imageSourceBoardIds = useMemo(() => [image.boardId], [image.boardId]);
  const imageItemRef = useMemo<GalleryItemRef>(() => ({ kind: 'image', name: image.imageName }), [image.imageName]);
  const handleSelectForCompare = useCallback(() => actions.selectForCompare(image), [actions, image]);
  const handleLoadWorkflow = useCallback(() => void actions.loadImageWorkflow(image), [actions, image]);
  const handleGeneratePbrMaps = useCallback(() => void actions.generatePbrMaps(image), [actions, image]);

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
      {variant === 'full' ? (
        <>
          <Menu.Separator borderColor="border.subtle" />
          <ImageContinueGroup actions={actions} images={images} presentation="compact" />
          <Menu.Separator borderColor="border.subtle" />
          <ImageUseGroups
            actions={actions}
            capabilities={recallCapabilities}
            image={image}
            isLoadingCapabilities={isLoadingRecallCapabilities}
            presentation="compact"
          />
          <Menu.Separator borderColor="border.subtle" />
        </>
      ) : null}
      <ContextMenuItem
        // The grid's shortcut for the same action, otherwise found only by accident.
        detail={t('widgets.gallery.itemMenu.compareShortcut', { key: formatHotkeyPartLabel('alt') })}
        icon={ImagesIcon}
        label={t('widgets.preview.selectForCompare')}
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
      <ChangeBoardSubMenu boards={boards} itemBoardIds={imageSourceBoardIds} onMove={handleMove} />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        isDanger
        icon={Trash2Icon}
        label={t('widgets.gallery.itemMenu.deleteImage')}
        value="delete-image"
        onClick={handleDelete}
      />
    </>
  );
};

const OpenInNewTabQuickMenuItem = ({ image }: { image: GalleryImage }) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => window.open(image.imageUrl, '_blank', 'noopener'), [image.imageUrl]);

  return (
    <MenuIconItem
      icon={ExternalLinkIcon}
      label={t('widgets.gallery.itemMenu.openInNewTab')}
      value="open-in-new-tab"
      onSelect={handleClick}
    />
  );
};

const CopyQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => void actions.copyImage(image), [actions, image]);

  return (
    <MenuIconItem
      icon={CopyIcon}
      label={t('widgets.gallery.itemMenu.copyToClipboard')}
      unavailableReason={canCopyImage() ? undefined : t('common.copyUnavailable')}
      value="copy-to-clipboard"
      onSelect={handleClick}
    />
  );
};

const DownloadQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => void actions.downloadImage(image), [actions, image]);

  return (
    <MenuIconItem
      icon={DownloadIcon}
      label={t('widgets.gallery.itemMenu.downloadImage')}
      value="download-image"
      onSelect={handleClick}
    />
  );
};

const OpenPreviewQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => actions.openImageInPreview(image), [actions, image]);

  return (
    <MenuIconItem
      icon={EyeIcon}
      label={t('widgets.gallery.itemMenu.openInPreview')}
      value="open-in-preview"
      onSelect={handleClick}
    />
  );
};

const ToggleStarQuickMenuItem = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const { t } = useTranslation();
  const handleClick = useCallback(
    () => void actions.setImagesStarred([image.imageName], !image.starred),
    [actions, image.imageName, image.starred]
  );

  return (
    <MenuIconItem
      icon={StarIcon}
      iconFill={image.starred ? 'currentColor' : 'none'}
      label={t(image.starred ? 'widgets.preview.unstarImage' : 'widgets.preview.starImage')}
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
  const { t } = useTranslation();
  const allStarred = images.every((image) => image.starred);
  const sourceBoardIds = useMemo(() => images.map((image) => image.boardId), [images]);
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
        {t('widgets.gallery.itemMenu.imagesSelected', { count: images.length })}
      </Text>
      <Menu.Separator borderColor="border.subtle" />
      <ImageContinueGroup actions={actions} images={images} presentation="compact" />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        icon={StarIcon}
        iconFill={allStarred ? 'currentColor' : 'none'}
        label={t(allStarred ? 'widgets.gallery.unstarSelection' : 'widgets.gallery.starSelection')}
        value="toggle-starred-all"
        onClick={handleToggleStarred}
      />
      <ContextMenuItem
        icon={DownloadIcon}
        label={t('widgets.gallery.downloadSelection')}
        value="download-selection"
        onClick={handleDownload}
      />
      <ChangeBoardSubMenu boards={boards} itemBoardIds={sourceBoardIds} onMove={handleMove} />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        isDanger
        icon={Trash2Icon}
        label={t('widgets.gallery.deleteSelection')}
        value="delete-selection"
        onClick={handleDelete}
      />
    </>
  );
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

/** The move targets the selection bar offers too (`getGalleryMoveTargets`), for the items' boards as known. */
const ChangeBoardSubMenu = ({
  boards,
  itemBoardIds,
  listingBoardId,
  unloadedCount = 0,
  onMove,
}: {
  boards: GalleryBoard[];
  itemBoardIds: readonly string[];
  listingBoardId?: string;
  unloadedCount?: number;
  onMove: (boardId: string) => void;
}) => {
  const { t } = useTranslation();
  const handleRemoveFromBoard = useCallback(() => onMove('none'), [onMove]);
  const handleSelectBoard = useCallback((board: GalleryBoard) => onMove(board.id), [onMove]);
  const { boards: destinations, canRemoveFromBoard } = useMemo(
    () => getGalleryMoveTargets(boards, { itemBoardIds, listingBoardId, unloadedCount }),
    [boards, itemBoardIds, listingBoardId, unloadedCount]
  );

  return (
    <ContextSubMenu icon={FolderIcon} label={t('widgets.gallery.changeBoard')}>
      {canRemoveFromBoard && (
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

const ContextMenuItem = ({
  detail,
  disabled,
  icon,
  iconFill,
  isDanger,
  label,
  unavailableReason,
  value,
  onClick,
}: {
  /** A short hint after the label, such as the pointer shortcut for the same action. */
  detail?: string;
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
      {detail ? (
        <Text color="fg.muted" flexShrink={0} fontSize="sm">
          {detail}
        </Text>
      ) : null}
    </HStack>
  </MenuItem>
);
