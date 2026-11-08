import type { GenerationSelectedImage } from '@features/generation/ui/GenerationUiContext';

import { useDndContext, useDndMonitor } from '@dnd-kit/core';
import { galleryImageUrls, isGalleryImageDragData, useGalleryImageDroppable } from '@features/gallery/utility';
import { useCallback, useId, useMemo, useState } from 'react';

const DROP_DATA = { kind: 'prompt-image' } as const;

export interface DroppedPromptImage {
  /** The dropped image, or null when the popover should follow the gallery selection. */
  image: GenerationSelectedImage | null;
  /** Hands the popover back to the gallery selection once it is done with the drop. */
  onClear: () => void;
}

export interface PromptImageDrop {
  droppedImage: DroppedPromptImage;
  /** A gallery image is mid-drag, so the drop affordance should be showing. */
  isDragActive: boolean;
  isOver: boolean;
  setNodeRef: (element: HTMLElement | null) => void;
}

/** Dropped images override gallery selection until consumed and cleared. */
export const usePromptImageDrop = ({ disabled = false }: { disabled?: boolean } = {}): PromptImageDrop => {
  const dropId = useId();
  const [image, setImage] = useState<GenerationSelectedImage | null>(null);
  const { active } = useDndContext();
  const { isOver, setNodeRef } = useGalleryImageDroppable({ data: DROP_DATA, disabled, id: dropId });

  useDndMonitor({
    onDragEnd: (event) => {
      const data = event.active.data.current;

      if (event.over?.id !== dropId || !isGalleryImageDragData(data)) {
        return;
      }

      // The popover describes one image, so a multi-image drag describes its first.
      const [first] = data.items;

      setImage({
        imageName: first.name,
        imageUrl: galleryImageUrls.full(first.name),
        thumbnailUrl: galleryImageUrls.thumbnail(first.name),
      });
    },
  });

  const onClear = useCallback(() => setImage(null), []);
  const droppedImage = useMemo(() => ({ image, onClear }), [image, onClear]);

  return {
    droppedImage,
    isDragActive: !disabled && isGalleryImageDragData(active?.data.current),
    isOver,
    setNodeRef,
  };
};
