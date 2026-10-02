import type { GalleryItem } from '@features/gallery';
import type { GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';
import type { ImageActions } from '@workbench/image-actions';

import { HStack, Icon, Menu, Portal } from '@chakra-ui/react';
import { galleryImageItemToGalleryImage, isGalleryImageItem, toGalleryItemRef } from '@features/gallery/contracts';
import { Button, IconButton, MenuActionItem, MenuContent, Tooltip, useTooltipTriggerIds } from '@platform/ui';
import { getGalleryCanvasImportMenuItems } from '@workbench/image-actions';
import {
  ChevronDownIcon,
  CopyIcon,
  EllipsisVerticalIcon,
  ImagesIcon,
  PencilIcon,
  StarIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewDensity } from './previewDensity';

const CANVAS_ITEMS = getGalleryCanvasImportMenuItems(false);
const CANVAS_MENU_POSITIONING = { placement: 'bottom-start' } as const;
const STAR_LABEL_KEYS = {
  image: { off: 'widgets.preview.starImage', on: 'widgets.preview.unstarImage' },
  video: { off: 'widgets.preview.starVideo', on: 'widgets.preview.unstarVideo' },
} as const;

export const PreviewActionStrip = ({
  actions,
  density,
  isVideoFrameCopyAvailable = false,
  item,
  onCopyCurrentFrame,
  onOpenMenu,
}: {
  actions: ImageActions;
  density: PreviewDensity;
  isVideoFrameCopyAvailable?: boolean;
  item: GalleryItem;
  onCopyCurrentFrame?: () => void;
  /** Opens the view's image context menu at viewport coordinates. */
  onOpenMenu: ((x: number, y: number) => void) | null;
}) => {
  const { t } = useTranslation();
  const image = isGalleryImageItem(item) ? galleryImageItemToGalleryImage(item) : null;
  const toggleStar = useCallback(
    () => void actions.setItemsStarred([toGalleryItemRef(item)], !item.starred),
    [actions, item]
  );
  const selectForCompare = useCallback(() => image && actions.selectForCompare(image), [actions, image]);
  const sendToCanvas = useCallback(
    (destination: GalleryCanvasImportDestination) => image && void actions.sendToCanvas([image], destination),
    [actions, image]
  );
  const copyCurrentFrame = useCallback(() => onCopyCurrentFrame?.(), [onCopyCurrentFrame]);
  const openMenu = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();

      onOpenMenu?.(rect.left, rect.bottom + 4);
    },
    [onOpenMenu]
  );
  const starLabel = t(STAR_LABEL_KEYS[item.kind][item.starred ? 'on' : 'off']);
  const starButton = (
    <Tooltip content={starLabel}>
      <IconButton aria-label={starLabel} color="fg.muted" size="2xs" variant="ghost" onClick={toggleStar}>
        <Icon as={StarIcon} boxSize="3.5" fill={item.starred ? 'currentColor' : 'none'} />
      </IconButton>
    </Tooltip>
  );
  const menuButton = onOpenMenu ? (
    <StripIconButton icon={EllipsisVerticalIcon} label={t('widgets.preview.imageActions')} onClick={openMenu} />
  ) : null;

  if (density !== 'full') {
    return (
      <HStack flexShrink={0} gap="0.5">
        {starButton}
        {menuButton}
      </HStack>
    );
  }

  return (
    <HStack flexShrink={0} gap="0.5">
      {/* Lead with the labeled canvas-destination action; omit it for videos, matching context-menu eligibility. */}
      {image ? (
        <>
          <EditOnCanvasMenu onSend={sendToCanvas} />
          <StripIconButton icon={ImagesIcon} label={t('widgets.preview.selectForCompare')} onClick={selectForCompare} />
        </>
      ) : null}
      {starButton}
      {item.kind === 'video' && onCopyCurrentFrame ? (
        <StripIconButton
          disabled={!isVideoFrameCopyAvailable}
          icon={CopyIcon}
          label={t('widgets.preview.copyCurrentFrame')}
          onClick={copyCurrentFrame}
        />
      ) : null}
      {menuButton}
    </HStack>
  );
};

/** "Edit" opens onto every canvas layer destination the context menu offers. */
const EditOnCanvasMenu = ({ onSend }: { onSend: (destination: GalleryCanvasImportDestination) => void }) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();

  return (
    <Menu.Root ids={ids} positioning={CANVAS_MENU_POSITIONING}>
      <Tooltip content={t('widgets.preview.editOnCanvas')} ids={ids}>
        <Menu.Trigger asChild>
          <Button aria-label={t('widgets.preview.editOnCanvas')} color="fg.muted" size="2xs" variant="ghost">
            <Icon as={PencilIcon} boxSize="3.5" />
            {t('common.edit')}
            <ChevronDownIcon size={12} />
          </Button>
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="14rem">
            {CANVAS_ITEMS.map((item) => (
              <CanvasDestinationItem
                key={item.destination}
                destination={item.destination}
                label={t(item.label)}
                value={item.value}
                onSend={onSend}
              />
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const CanvasDestinationItem = ({
  destination,
  label,
  onSend,
  value,
}: {
  destination: GalleryCanvasImportDestination;
  label: string;
  onSend: (destination: GalleryCanvasImportDestination) => void;
  value: string;
}) => {
  const handleSelect = useCallback(() => onSend(destination), [destination, onSend]);

  return <MenuActionItem label={label} value={value} onSelect={handleSelect} />;
};

const StripIconButton = ({
  disabled,
  icon,
  label,
  onClick,
}: {
  disabled?: boolean;
  icon: LucideIcon;
  label: string;
  onClick: (event: MouseEvent<HTMLButtonElement>) => void;
}) => (
  <Tooltip content={label}>
    <IconButton aria-label={label} color="fg.muted" disabled={disabled} size="2xs" variant="ghost" onClick={onClick}>
      <Icon as={icon} boxSize="3.5" />
    </IconButton>
  </Tooltip>
);
