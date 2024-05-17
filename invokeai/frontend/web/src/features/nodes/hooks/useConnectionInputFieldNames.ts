import { EMPTY_ARRAY } from 'app/store/constants';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { TEMPLATE_BUILDER_MAP } from 'features/nodes/util/schema/buildFieldInputTemplate';
import { keys, map } from 'lodash-es';
import { useMemo } from 'react';

export const useConnectionInputFieldNames = (nodeId: string): string[] => {
  const template = useNodeTemplate(nodeId);
  const fieldNames = useMemo(() => {
    // get the visible fields
    const fields = map(template.inputs).filter(
      (field) =>
        (field.input === 'connection' && !field.type.isCollectionOrScalar) ||
        !keys(TEMPLATE_BUILDER_MAP).includes(field.type.name)
    );

    const _fieldNames = getSortedFilteredFieldNames(fields);

    if (_fieldNames.length === 0) {
      return EMPTY_ARRAY;
    }

    return _fieldNames;
  }, [template.inputs]);

  return fieldNames;
};
