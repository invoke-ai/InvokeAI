import type { GalleryImage } from '@workbench/gallery/api';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { IconButton, MenuContent, Tooltip } from '@workbench/components/ui';
import { getGalleryCanvasImportMenuItems, type ImageActions } from '@workbench/image-actions';
import {
  CopyIcon,
  DownloadIcon,
  ImagesIcon,
  LayersIcon,
  MoreHorizontalIcon,
  StarIcon,
  Trash2Icon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewDensity } from './previewDensity';

/**
 * The footer's always-visible action row: the most-used image verbs, one click
 * away instead of buried in the context menu. At full density every action is
 * an icon button; compact/minimal collapse to star + an overflow menu.
 */

const MENU_POSITIONING = { placement: 'bottom-end' } as const;

export const PreviewActionStrip = ({
  actions,
  density,
  image,
}: {
  actions: ImageActions;
  density: PreviewDensity;
  image: GalleryImage;
}) => {
  const { t } = useTranslation();
  const canvasImportItems = useMemo(() => getGalleryCanvasImportMenuItems(false), []);
  const toggleStar = useCallback(
    () => void actions.setImagesStarred([image.imageName], !image.starred),
    [actions, image.imageName, image.starred]
  );
  const selectForCompare = useCallback(() => actions.selectForCompare(image), [actions, image]);
  const copyImage = useCallback(() => void actions.copyImage(image), [actions, image]);
  const downloadImage = useCallback(() => void actions.downloadImage(image), [actions, image]);
  const deleteImage = useCallback(() => void actions.deleteImages([image.imageName]), [actions, image.imageName]);
  const sendToCanvas = useCallback(
    (destination: (typeof canvasImportItems)[number]['destination']) => void actions.sendToCanvas([image], destination),
    [actions, image]
  );
  const starLabel = image.starred ? 'Unstar image' : 'Star image';
  const starButton = (
    <Tooltip content={starLabel}>
      <IconButton aria-label={starLabel} color="fg.muted" size="2xs" variant="ghost" onClick={toggleStar}>
        <Icon as={StarIcon} boxSize="3.5" fill={image.starred ? 'currentColor' : 'none'} />
      </IconButton>
    </Tooltip>
  );

  if (density !== 'full') {
    return (
      <HStack flexShrink={0} gap="0.5">
        {starButton}
        <Menu.Root positioning={MENU_POSITIONING}>
          <Menu.Trigger asChild>
            <IconButton aria-label="Image actions" color="fg.muted" size="2xs" variant="ghost">
              <Icon as={MoreHorizontalIcon} boxSize="3.5" />
            </IconButton>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="12rem">
                <StripMenuItem
                  icon={ImagesIcon}
                  label="Select for Compare"
                  value="compare"
                  onClick={selectForCompare}
                />
                <StripMenuItem icon={CopyIcon} label="Copy to clipboard" value="copy" onClick={copyImage} />
                <StripMenuItem icon={DownloadIcon} label="Download image" value="download" onClick={downloadImage} />
                <Menu.Separator borderColor="border.subtle" />
                {canvasImportItems.map((item) => (
                  <StripCanvasImportMenuItem
                    key={item.value}
                    destination={item.destination}
                    label={t(item.label)}
                    value={item.value}
                    onSend={sendToCanvas}
                  />
                ))}
                <Menu.Separator borderColor="border.subtle" />
                <StripMenuItem
                  color="fg.error"
                  icon={Trash2Icon}
                  label="Delete Image"
                  value="delete"
                  onClick={deleteImage}
                />
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </HStack>
    );
  }

  return (
    <HStack flexShrink={0} gap="0.5">
      {starButton}
      <StripIconButton icon={ImagesIcon} label="Select for Compare" onClick={selectForCompare} />
      <StripIconButton icon={CopyIcon} label="Copy to clipboard" onClick={copyImage} />
      <StripIconButton icon={DownloadIcon} label="Download image" onClick={downloadImage} />
      <Menu.Root positioning={MENU_POSITIONING}>
        <Menu.Trigger asChild>
          <IconButton aria-label="Send to canvas" color="fg.muted" size="2xs" variant="ghost">
            <Icon as={LayersIcon} boxSize="3.5" />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="12rem">
              {canvasImportItems.map((item) => (
                <StripCanvasImportMenuItem
                  key={item.value}
                  destination={item.destination}
                  label={t(item.label)}
                  value={item.value}
                  onSend={sendToCanvas}
                />
              ))}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <StripIconButton icon={Trash2Icon} label="Delete Image" onClick={deleteImage} />
    </HStack>
  );
};

const StripIconButton = ({ icon, label, onClick }: { icon: LucideIcon; label: string; onClick: () => void }) => (
  <Tooltip content={label}>
    <IconButton aria-label={label} color="fg.muted" size="2xs" variant="ghost" onClick={onClick}>
      <Icon as={icon} boxSize="3.5" />
    </IconButton>
  </Tooltip>
);

const StripMenuItem = ({
  color,
  icon,
  label,
  value,
  onClick,
}: {
  color?: string;
  icon: LucideIcon;
  label: string;
  value: string;
  onClick: () => void;
}) => (
  <Menu.Item color={color} value={value} onClick={onClick}>
    <Icon as={icon} boxSize="3.5" color={color ?? 'fg.subtle'} />
    <Text fontSize="xs">{label}</Text>
  </Menu.Item>
);

const StripCanvasImportMenuItem = ({
  destination,
  label,
  value,
  onSend,
}: {
  destination: Parameters<ImageActions['sendToCanvas']>[1];
  label: string;
  value: string;
  onSend: (destination: Parameters<ImageActions['sendToCanvas']>[1]) => void;
}) => {
  const handleClick = useCallback(() => onSend(destination), [destination, onSend]);

  return <StripMenuItem icon={LayersIcon} label={label} value={value} onClick={handleClick} />;
};
