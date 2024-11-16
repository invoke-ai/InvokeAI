import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { ImageProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Image } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { singleImageDndSource } from 'features/dnd/dnd';
import type { DndDragPreviewSingleImageState } from 'features/dnd/DndDragPreviewSingleImage';
import { createSingleImageDragPreview, setSingleImageDragPreview } from 'features/dnd/DndDragPreviewSingleImage';
import { firefoxDndFix } from 'features/dnd/util';
import { useImageContextMenu } from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import { memo, useEffect, useState } from 'react';
import type { ImageDTO } from 'services/api/types';

const sx = {
  objectFit: 'contain',
  maxW: 'full',
  maxH: 'full',
  borderRadius: 'base',
  cursor: 'grab',
  '&[data-is-dragging=true]': {
    opacity: 0.3,
  },
} satisfies SystemStyleObject;

/* eslint-disable-next-line @typescript-eslint/no-namespace */
export namespace DndImage {
  export interface Props extends ImageProps {
    imageDTO: ImageDTO;
    asThumbnail?: boolean;
  }
}

export const DndImage = memo(({ imageDTO, asThumbnail, ...rest }: DndImage.Props) => {
  const store = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [element, ref] = useState<HTMLImageElement | null>(null);
  const [dragPreviewState, setDragPreviewState] = useState<DndDragPreviewSingleImageState | null>(null);

  useEffect(() => {
    if (!element) {
      return;
    }
    return combine(
      firefoxDndFix(element),
      draggable({
        element,
        getInitialData: () => singleImageDndSource.getData({ imageDTO }, imageDTO.image_name),
        onDragStart: () => {
          setIsDragging(true);
        },
        onDrop: () => {
          setIsDragging(false);
        },
        onGenerateDragPreview: (args) => {
          if (singleImageDndSource.typeGuard(args.source.data)) {
            setSingleImageDragPreview({
              singleImageDndData: args.source.data,
              onGenerateDragPreviewArgs: args,
              setDragPreviewState,
            });
          }
        },
      })
    );
  }, [imageDTO, element, store]);

  useImageContextMenu(imageDTO, element);

  return (
    <>
      <Image
        role="button"
        ref={ref}
        src={asThumbnail ? imageDTO.thumbnail_url : imageDTO.image_url}
        fallbackSrc={asThumbnail ? undefined : imageDTO.thumbnail_url}
        w={imageDTO.width}
        sx={sx}
        data-is-dragging={isDragging}
        {...rest}
      />
      {dragPreviewState?.type === 'single-image' ? createSingleImageDragPreview(dragPreviewState) : null}
    </>
  );
});

DndImage.displayName = 'DndImage';
