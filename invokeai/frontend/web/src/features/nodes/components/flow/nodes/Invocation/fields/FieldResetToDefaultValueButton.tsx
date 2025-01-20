import { FieldResetValueButton } from 'features/nodes/components/flow/nodes/Invocation/fields/FieldResetValueButton';
import { useFieldDefaultValue } from 'features/nodes/hooks/useFieldDefaultValue';
import { memo } from 'react';

type Props = {
  nodeId: string;
  fieldName: string;
};

const FieldResetToDefaultValueButton = ({ nodeId, fieldName }: Props) => {
  const { isValueChanged, resetToDefaultValue } = useFieldDefaultValue(nodeId, fieldName);

  return <FieldResetValueButton onClick={resetToDefaultValue} isDisabled={!isValueChanged} />;
};

export default memo(FieldResetToDefaultValueButton);
