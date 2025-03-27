import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { map } from 'lodash-es';
import { useMemo } from 'react';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useOutputFieldNames = (nodeId: string): string[] => {
  const template = useNodeTemplateOrThrow(nodeId);
  const fieldNames = useMemo(() => getSortedFilteredFieldNames(map(template.outputs)), [template.outputs]);
  return fieldNames;
};
