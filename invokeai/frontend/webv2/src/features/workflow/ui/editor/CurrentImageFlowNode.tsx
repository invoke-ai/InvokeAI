import type { GeneratedImageContract } from '@features/gallery';
import type { GalleryItemRef } from '@features/gallery/contracts';
import type { NodeProps } from '@xyflow/react';
import type { MouseEvent, ReactNode } from 'react';

import { Box, Flex, Icon, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getSelectedGalleryImageFromValues } from '@features/gallery/contracts';
import { useGalleryItemContextMenu } from '@features/gallery/react';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { useProgressImage } from '@features/queue/react';
import {
  getWorkflowNodeHeaderProps,
  getWorkflowNodeShellProps,
  useIsWorkflowImageExport,
  WorkflowNodeHeaderButton,
} from '@features/workflow/ui/nodeChrome';
import { useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import {
  imageUrlToStreamingSource,
  progressImageToStreamingSource,
} from '@platform/ui/streaming-image/streamingImageSource';
import { EllipsisVerticalIcon } from 'lucide-react';
import { memo, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { CurrentImageFlowNode as CurrentImageFlowNodeType } from './flowAdapters';

/** What the node shows when no run is drawing: the image the gallery has selected, as legacy did. */
interface CurrentImageView {
  height?: number;
  imageName: string;
  imageUrl: string;
  width?: number;
}

/**
 * The gallery's selected image. A selected video is passed over, as legacy did; with no image selected the node falls
 * back to the project's most recent image, so it is not empty while results exist.
 */
export const getCurrentImageView = (galleryValues: Record<string, unknown>): CurrentImageView | null => {
  const selected = getSelectedGalleryImageFromValues(galleryValues);
  const recentImages = Array.isArray(galleryValues.recentImages)
    ? (galleryValues.recentImages as GeneratedImageContract[])
    : [];
  const image = selected ?? recentImages[0];

  return image
    ? { height: image.height, imageName: image.imageName, imageUrl: image.imageUrl, width: image.width }
    : null;
};

const isSameImageView = (a: CurrentImageView | null, b: CurrentImageView | null): boolean =>
  a === b ||
  (a !== null &&
    b !== null &&
    a.imageName === b.imageName &&
    a.imageUrl === b.imageUrl &&
    a.width === b.width &&
    a.height === b.height);

/** The image under the node's frame, draggable wherever gallery images can be dropped. */
const DraggableCurrentImage = ({
  children,
  itemRef,
  onContextMenu,
}: {
  children: ReactNode;
  itemRef: GalleryItemRef | null;
  onContextMenu: ((event: MouseEvent<HTMLElement>) => void) | undefined;
}) => {
  const disabledId = useId();
  const dragData = useMemo(() => (itemRef ? getGalleryItemDragData([itemRef]) : undefined), [itemRef]);
  // Pointer drags only, as the Preview frame's: the image is no tab stop, so the canvas keeps its own keys. The header's
  // button is the keyboard's way to the item menu.
  const { listeners, setNodeRef } = useDraggable({
    data: dragData,
    disabled: !itemRef,
    id: itemRef
      ? getGalleryItemDragId(itemRef, 'workflow-current-image', disabledId)
      : `workflow-current-image:disabled:${disabledId}`,
  });

  return (
    // `nodrag`: the canvas leaves the pointer to the image drag rather than moving the node.
    <Box
      ref={setNodeRef}
      className="nodrag"
      cursor={itemRef ? 'grab' : undefined}
      onContextMenu={onContextMenu}
      {...listeners}
    >
      {children}
    </Box>
  );
};

const CurrentImageEditorNode = ({ data, selected }: NodeProps<CurrentImageFlowNodeType>) => {
  const { t } = useTranslation();
  // Only the shown image: other gallery changes (filters, scroll, other selections' metadata) do not render the node.
  const image = useWorkflowProjectSelector((project) => getCurrentImageView(project.galleryValues), isSameImageView);
  const progressImage = useProgressImage();
  const node = data.documentNode;
  const title = node.data.label || t('widgets.workflow.addNodeDialog.currentImageTitle');
  const shownImageName = progressImage ? null : image?.imageName;
  const itemRef = useMemo<GalleryItemRef | null>(
    () => (shownImageName ? { kind: 'image', name: shownImageName } : null),
    [shownImageName]
  );
  const openItemMenu = useGalleryItemContextMenu(itemRef);
  const imageActionsLabel = t('widgets.preview.imageActions');

  return (
    <Box
      data-is-selected={selected}
      data-workflow-node-shell="true"
      overflow="hidden"
      w="20rem"
      {...getWorkflowNodeShellProps({ selected })}
    >
      <Flex {...getWorkflowNodeHeaderProps({ endsWithButton: openItemMenu !== undefined })}>
        <MiddleTruncate flex="1" fontWeight="600" minW="0" text={title} />
        {progressImage ? (
          <Text color="brand.solid" data-node-status-indicator="true" flexShrink={0} fontSize="xs">
            {t('widgets.workflow.generating')}
          </Text>
        ) : openItemMenu ? (
          <WorkflowNodeHeaderButton
            aria-haspopup="menu"
            aria-label={imageActionsLabel}
            className="nodrag"
            content={imageActionsLabel}
            onClick={openItemMenu}
          >
            <Icon as={EllipsisVerticalIcon} boxSize="3.5" />
          </WorkflowNodeHeaderButton>
        ) : null}
      </Flex>
      <DraggableCurrentImage itemRef={itemRef} onContextMenu={openItemMenu}>
        <StreamingImageFrame
          fallbackImage={imageUrlToStreamingSource({
            alt: image?.imageName ?? title,
            height: image?.height,
            kind: 'fallback',
            src: image?.imageUrl,
            width: image?.width,
          })}
          fit="contain"
          h="18rem"
          liveImage={progressImageToStreamingSource(progressImage)}
          w="full"
        >
          <Flex align="center" color="fg.subtle" fontSize="xs" h="full" justify="center" px="4" textAlign="center">
            {t('widgets.workflow.currentImageEmpty')}
          </Flex>
        </StreamingImageFrame>
      </DraggableCurrentImage>
    </Box>
  );
};

const CurrentImageSnapshotNode = ({ data, selected }: NodeProps<CurrentImageFlowNodeType>) => {
  const { t } = useTranslation();
  const node = data.documentNode;

  return (
    <Box
      data-is-selected={selected}
      data-workflow-node-shell="true"
      w="20rem"
      {...getWorkflowNodeShellProps({ selected })}
    >
      <Flex {...getWorkflowNodeHeaderProps()}>
        <MiddleTruncate
          data-workflow-export-node-title="true"
          data-workflow-export-static-node-content="true"
          flex="1"
          fontWeight="600"
          minW="0"
          text={node.data.label || t('widgets.workflow.addNodeDialog.currentImageTitle')}
        />
      </Flex>
    </Box>
  );
};

const CurrentImageFlowNodeComponent = (props: NodeProps<CurrentImageFlowNodeType>) => {
  const isWorkflowImageExport = useIsWorkflowImageExport();

  return isWorkflowImageExport ? <CurrentImageSnapshotNode {...props} /> : <CurrentImageEditorNode {...props} />;
};

export const CurrentImageFlowNode = memo(CurrentImageFlowNodeComponent);
