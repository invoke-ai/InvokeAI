import { useNodeTemplateOrThrow } from 'features/nodes/hooks/useNodeTemplateOrThrow';
import { getHasNodeFooter } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

/**
 * Whether the node renders a footer. The footer hosts the node attribute fields and their connection handles.
 *
 * @see {@link getHasNodeFooter}
 */
export const useWithFooter = () => {
  const template = useNodeTemplateOrThrow();
  return useMemo(() => getHasNodeFooter(template), [template]);
};
