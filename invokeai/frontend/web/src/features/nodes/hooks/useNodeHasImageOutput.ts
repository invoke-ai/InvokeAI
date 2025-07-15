import { some } from 'es-toolkit/compat';
import { useMemo } from 'react';

import { useNodeTemplateSafe } from './useNodeTemplateSafe';

export const useNodeHasImageOutput = (): boolean => {
  const template = useNodeTemplateSafe();
  const hasImageOutput = useMemo(
    () =>
      some(
        template?.outputs,
        (output) =>
          output.type.name === 'ImageField' &&
          // the image primitive node (node type "image") does not actually save the image, do not show the image-saving checkboxes
          template?.type !== 'image'
      ),
    [template]
  );

  return hasImageOutput;
};
