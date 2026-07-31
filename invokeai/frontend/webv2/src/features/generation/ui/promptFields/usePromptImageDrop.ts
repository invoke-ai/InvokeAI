import type { GenerationSelectedImage } from '@features/generation/ui/GenerationUiContext';

import { useDndContext, useDndMonitor } from '@dnd-kit/core';
import { galleryImageUrls, isGalleryImageDragData, useGalleryImageDroppable } from '@features/gallery/utility';
import { useCallback, useId, useMemo, useState } from 'react';

const DROP_DATA = { kind: 'prompt-image' } as const;

/**
 * An image handed to the prompt box by a drag, waiting for the image-to-prompt
 * popover to pick it up.
 */
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

/**
 * Makes an element a drop target for a gallery image and holds what landed on
 * it. Dropping is the direct way to describe an image that is not the gallery's
 * current selection — the popover reads this in preference to that selection,
 * until it is done and clears it.
 */
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
