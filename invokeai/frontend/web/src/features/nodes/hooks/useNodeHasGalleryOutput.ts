import { some } from 'es-toolkit/compat';
import { useMemo } from 'react';

import { useNodeTemplateSafe } from './useNodeTemplateSafe';

/**
 * True when the node produces an output that lands in the gallery — currently ImageField or
 * VideoField. Used to gate the "Save in gallery" checkbox and the footer that contains it.
 *
 * The `image` primitive node is excluded because it passes through an existing image without
 * saving a new one; no equivalent video primitive exists yet.
 */
export const useNodeHasGalleryOutput = (): boolean => {
  const template = useNodeTemplateSafe();
  const hasGalleryOutput = useMemo(
    () =>
      some(
        template?.outputs,
        (output) =>
          (output.type.name === 'ImageField' && template?.type !== 'image') || output.type.name === 'VideoField'
      ),
    [template]
  );

  return hasGalleryOutput;
};
