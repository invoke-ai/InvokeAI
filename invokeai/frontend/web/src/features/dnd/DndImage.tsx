import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { ImageProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Image } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $crossOrigin } from 'app/store/nanostores/authToken';
import { useAppStore } from 'app/store/storeHooks';
import { singleImageDndSource } from 'features/dnd/dnd';
import type { DndDragPreviewSingleImageState } from 'features/dnd/DndDragPreviewSingleImage';
import { createSingleImageDragPreview, setSingleImageDragPreview } from 'features/dnd/DndDragPreviewSingleImage';
import { firefoxDndFix } from 'features/dnd/util';
import { Editor } from 'features/editImageModal/lib/editor';
import { openEditImageModal } from 'features/editImageModal/store';
import { useImageContextMenu } from 'features/gallery/components/ContextMenu/ImageContextMenu';
import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import type { ImageDTO } from 'services/api/types';

const sx = {
  objectFit: 'contain',
  maxW: 'full',
  maxH: 'full',
  cursor: 'grab',
  '&[data-is-dragging=true]': {
    opacity: 0.3,
  },
} satisfies SystemStyleObject;

type Props = {
  imageDTO: ImageDTO;
  asThumbnail?: boolean;
  editable?: boolean;
} & ImageProps;

export const DndImage = memo(
  forwardRef(({ imageDTO, asThumbnail, editable, ...rest }: Props, forwardedRef) => {
    const store = useAppStore();
    const crossOrigin = useStore($crossOrigin);
    const [previewDataURL, setPreviewDataURl] = useState<string | null>(null);

    const [isDragging, setIsDragging] = useState(false);
    const ref = useRef<HTMLImageElement>(null);
    useImperativeHandle(forwardedRef, () => ref.current!, []);
    const [dragPreviewState, setDragPreviewState] = useState<DndDragPreviewSingleImageState | null>(null);

    useEffect(() => {
      const element = ref.current;
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
    }, [forwardedRef, imageDTO, store]);

    useImageContextMenu(imageDTO, ref);

    const edit = useCallback(() => {
      if (!editable) {
        return;
      }

      const editor = new Editor();
      editor.onCropApply(async () => {
        const previewDataURL = await editor.exportImage('dataURL', { withCropOverlay: true });
        setPreviewDataURl(previewDataURL);
      });
      openEditImageModal(imageDTO.image_name, editor);
    }, [editable, imageDTO.image_name]);

    return (
      <>
        <Image
          role="button"
          ref={ref}
          src={previewDataURL ?? (asThumbnail ? imageDTO.thumbnail_url : imageDTO.image_url)}
          fallbackSrc={asThumbnail ? undefined : imageDTO.thumbnail_url}
          width={imageDTO.width}
          height={imageDTO.height}
          sx={sx}
          data-is-dragging={isDragging}
          crossOrigin={!asThumbnail ? crossOrigin : undefined}
          onClick={edit}
          {...rest}
        />
        {dragPreviewState?.type === 'single-image' ? createSingleImageDragPreview(dragPreviewState) : null}
      </>
    );
  })
);
DndImage.displayName = 'DndImage';
