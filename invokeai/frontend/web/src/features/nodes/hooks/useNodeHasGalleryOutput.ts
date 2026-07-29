import { some } from 'es-toolkit/compat';
import { useMemo } from 'react';

import { useNodeTemplateSafe } from './useNodeTemplateSafe';

/**
 * True when the node produces an output that lands in the gallery — currently ImageField or
 * VideoField. Used to gate the "Save in gallery" checkbox and the footer that contains it.
 *
 * The `image` and `video` primitive nodes are excluded because they pass through an existing
 * asset without saving a new copy.
 */
export const useNodeHasGalleryOutput = (): boolean => {
  const template = useNodeTemplateSafe();
  const hasGalleryOutput = useMemo(
    () =>
      some(
        template?.outputs,
        (output) =>
          (output.type.name === 'ImageField' && template?.type !== 'image') ||
          (output.type.name === 'VideoField' && template?.type !== 'video')
      ),
    [template]
  );

  return hasGalleryOutput;
};
