import { some } from 'lodash-es';
import { useMemo } from 'react';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useNodeHasImageOutput = (nodeId: string): boolean => {
  const template = useNodeTemplateOrThrow(nodeId);
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
