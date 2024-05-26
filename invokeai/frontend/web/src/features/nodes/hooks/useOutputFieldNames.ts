import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { map } from 'lodash-es';
import { useMemo } from 'react';

export const useOutputFieldNames = (nodeId: string): string[] => {
  const template = useNodeTemplate(nodeId);
  const fieldNames = useMemo(() => getSortedFilteredFieldNames(map(template.outputs)), [template.outputs]);
  return fieldNames;
};
