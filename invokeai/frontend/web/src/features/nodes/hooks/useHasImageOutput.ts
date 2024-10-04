import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { some } from 'lodash-es';
import { useMemo } from 'react';

export const useHasImageOutput = (nodeId: string): boolean => {
  const template = useNodeTemplate(nodeId);
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
