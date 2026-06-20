import type { GalleryBoard, GalleryImage } from '@workbench/gallery/api';

import { Dialog, HStack, Icon, Menu, Portal, ScrollArea, Text } from '@chakra-ui/react';
import { Button, MenuContent, Tooltip } from '@workbench/components/ui';
import { useWorkbenchPreferences } from '@workbench/settings/store';
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
import { useEffect, useRef, useState, type ReactNode } from 'react';

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

const isUsableGalleryImage = (value: unknown): value is GalleryImage =>
  Boolean(value) &&
  typeof value === 'object' &&
  typeof (value as GalleryImage).imageName === 'string' &&
  typeof (value as GalleryImage).imageUrl === 'string' &&
  typeof (value as GalleryImage).thumbnailUrl === 'string';

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
  const { confirmImageDeletion } = useWorkbenchPreferences();
  const [pendingDeletion, setPendingDeletion] = useState<string[] | null>(null);
  const [recallCapabilities, setRecallCapabilities] = useState<ImageRecallCapabilities>(
    EMPTY_IMAGE_RECALL_CAPABILITIES
  );
  const [isLoadingRecallCapabilities, setIsLoadingRecallCapabilities] = useState(false);
  const targetRef = useRef(target);
  const imageRef = useRef<GalleryImage | null>(null);

  targetRef.current = target;

  const isBulk = images.length > 1;
  const imageNames = images.map((candidate) => candidate.imageName);
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

  const requestDeletion = (names: string[]) => {
    if (confirmImageDeletion) {
      setPendingDeletion(names);
    } else {
      void actions.deleteImages(names);
    }
  };

  return (
    <>
      <Menu.Root
        lazyMount
        open
        positioning={{
          getAnchorRect: () => {
            const currentTarget = targetRef.current;

            return { height: 1, width: 1, x: currentTarget.x, y: currentTarget.y };
          },
          placement: 'bottom-start',
        }}
        unmountOnExit
        onOpenChange={(event) => {
          if (!event.open) {
            onClose();
          }
        }}
      >
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
                  isLoadingRecallCapabilities={isLoadingRecallCapabilities}
                  onRequestDeletion={requestDeletion}
                  recallCapabilities={recallCapabilities}
                />
              )}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Dialog.Root
        open={pendingDeletion !== null}
        role="alertdialog"
        onOpenChange={(event) => {
          if (!event.open) {
            setPendingDeletion(null);
          }
        }}
      >
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
                <Button size="xs" variant="outline" onClick={() => setPendingDeletion(null)}>
                  Cancel
                </Button>
                <Button
                  colorPalette="red"
                  size="xs"
                  onClick={() => {
                    if (pendingDeletion) {
                      void actions.deleteImages(pendingDeletion);
                    }

                    setPendingDeletion(null);
                  }}
                >
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
  isLoadingRecallCapabilities,
  onRequestDeletion,
  recallCapabilities,
}: {
  actions: ImageActions;
  boards: GalleryBoard[];
  image: GalleryImage;
  isLoadingRecallCapabilities: boolean;
  onRequestDeletion: (imageNames: string[]) => void;
  recallCapabilities: ImageRecallCapabilities;
}) => (
  <>
    <HStack gap="1">
      <QuickMenuItem
        icon={ExternalLinkIcon}
        label="Open in new tab"
        value="open-in-new-tab"
        onClick={() => window.open(image.imageUrl, '_blank', 'noopener')}
      />
      <QuickMenuItem
        icon={CopyIcon}
        label="Copy to clipboard"
        value="copy-to-clipboard"
        onClick={() => void actions.copyImage(image)}
      />
      <QuickMenuItem
        icon={DownloadIcon}
        label="Download image"
        value="download-image"
        onClick={() => void actions.downloadImage(image)}
      />
      <QuickMenuItem
        icon={EyeIcon}
        label="Open in preview"
        value="open-in-preview"
        onClick={() => actions.openImageInPreview(image)}
      />
      <QuickMenuItem
        icon={StarIcon}
        label={image.starred ? 'Unstar image' : 'Star image'}
        value="toggle-starred"
        onClick={() => void actions.setImagesStarred([image.imageName], !image.starred)}
      />
    </HStack>
    <Menu.Separator borderColor="border.subtle" />
    <ContextMenuItem disabled icon={WorkflowIcon} label="Load Workflow" value="load-workflow" />
    <ContextSubMenu icon={AsteriskIcon} label="Recall Metadata">
      <ContextMenuItem
        disabled={isLoadingRecallCapabilities || !recallCapabilities.all}
        icon={AsteriskIcon}
        label="Recall All"
        value="recall-all"
        onClick={() => void actions.recallImageData(image, 'all')}
      />
      <ContextMenuItem
        disabled={isLoadingRecallCapabilities || !recallCapabilities.remix}
        icon={ShuffleIcon}
        label="Remix Image"
        value="remix"
        onClick={() => void actions.recallImageData(image, 'remix')}
      />
      <ContextMenuItem
        disabled={isLoadingRecallCapabilities || !recallCapabilities.prompts}
        icon={QuoteIcon}
        label="Use Prompt"
        value="use-prompt"
        onClick={() => void actions.recallImageData(image, 'prompts')}
      />
      <ContextMenuItem
        disabled={isLoadingRecallCapabilities || !recallCapabilities.seed}
        icon={SproutIcon}
        label="Use Seed"
        value="use-seed"
        onClick={() => void actions.recallImageData(image, 'seed')}
      />
      <ContextMenuItem
        disabled={isLoadingRecallCapabilities || !recallCapabilities.dimensions}
        icon={RulerIcon}
        label="Use Size"
        value="use-size"
        onClick={() => void actions.recallImageData(image, 'dimensions')}
      />
      <ContextMenuItem
        disabled={isLoadingRecallCapabilities || !recallCapabilities.clipSkip}
        icon={ScissorsIcon}
        label="Use CLIP Skip"
        value="use-clip-skip"
        onClick={() => void actions.recallImageData(image, 'clipSkip')}
      />
    </ContextSubMenu>
    <Menu.Separator borderColor="border.subtle" />
    <ContextMenuItem disabled icon={ScanIcon} label="Send to Upscale" value="send-to-upscale" />
    <ContextMenuItem disabled icon={ImageIcon} label="Use as Reference Image" value="use-as-reference-image" />
    <ContextMenuItem disabled icon={TypeIcon} label="Use as Prompt Template" value="use-as-prompt-template" />
    <ContextMenuItem
      icon={ImagesIcon}
      label="Select for Compare"
      value="select-for-compare"
      onClick={() => actions.selectForCompare(image)}
    />
    <Menu.Separator borderColor="border.subtle" />
    <ContextSubMenu icon={FileImageIcon} label="New from Image">
      <ContextMenuItem disabled icon={FileImageIcon} label="New Canvas from Image" value="new-canvas-from-image" />
      <ContextMenuItem disabled icon={LayersIcon} label="New Layer from Image" value="new-layer-from-image" />
    </ContextSubMenu>
    <ChangeBoardSubMenu
      boards={boards}
      currentBoardId={image.boardId}
      onMove={(boardId) => void actions.moveImagesToBoard([image.imageName], boardId)}
    />
    <Menu.Separator borderColor="border.subtle" />
    <ContextMenuItem
      color="fg.error"
      icon={Trash2Icon}
      label="Delete Image"
      value="delete-image"
      onClick={() => onRequestDeletion([image.imageName])}
    />
  </>
);

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
        onClick={() => void actions.setImagesStarred(imageNames, !allStarred)}
      />
      <ContextMenuItem
        icon={DownloadIcon}
        label="Download Selection"
        value="download-selection"
        onClick={() => void actions.downloadImages(imageNames)}
      />
      <ChangeBoardSubMenu
        boards={boards}
        currentBoardId={null}
        onMove={(boardId) => void actions.moveImagesToBoard(imageNames, boardId)}
      />
      <Menu.Separator borderColor="border.subtle" />
      <ContextMenuItem
        color="fg.error"
        icon={Trash2Icon}
        label="Delete Selection"
        value="delete-selection"
        onClick={() => onRequestDeletion(imageNames)}
      />
    </>
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
}) => (
  <ContextSubMenu icon={FolderIcon} label="Change Board" scrollArea>
    {currentBoardId !== 'none' && (
      <ContextMenuItem
        icon={FolderIcon}
        label="Remove from Board"
        value="remove-from-board"
        onClick={() => onMove('none')}
      />
    )}
    {boards
      .filter((board) => board.kind === 'board' && board.id !== currentBoardId)
      .map((board) => (
        <Menu.Item key={board.id} value={`move-to-${board.id}`} onClick={() => onMove(board.id)}>
          <Text fontSize="xs" minW="0" truncate>
            {board.name}
          </Text>
        </Menu.Item>
      ))}
  </ContextSubMenu>
);

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
  <Menu.Root positioning={{ placement: 'right-start' }}>
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
    contentProps={{ fontSize: '2xs' }}
    openDelay={300}
    positioning={{ placement: 'top' }}
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
