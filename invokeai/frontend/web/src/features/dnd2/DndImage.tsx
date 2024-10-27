import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { ImageProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Image } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { singleImageDndSource } from 'features/dnd2/types';
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

type Props = ImageProps & {
  imageDTO: ImageDTO;
};

export const DndImage = memo(({ imageDTO, ...rest }: Props) => {
  const store = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [element, ref] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!element) {
      return;
    }
    return draggable({
      element,
      getInitialData: () => singleImageDndSource.getData({ imageDTO }, imageDTO.image_name),
      onDragStart: () => {
        setIsDragging(true);
      },
      onDrop: () => {
        setIsDragging(false);
      },
    });
  }, [imageDTO, element, store]);

  useImageContextMenu(imageDTO, element);

  return (
    <Image
      role="button"
      ref={ref}
      src={imageDTO.image_url}
      fallbackSrc={imageDTO.thumbnail_url}
      w={imageDTO.width}
      sx={sx}
      data-is-dragging={isDragging}
      {...rest}
    />
  );
});

DndImage.displayName = 'DndImage';
