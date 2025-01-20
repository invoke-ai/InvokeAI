import { FieldResetValueButton } from 'features/nodes/components/flow/nodes/Invocation/fields/FieldResetValueButton';
import { useFieldInitialLinearViewValue } from 'features/nodes/hooks/useFieldInitialLinearViewValue';
import { memo } from 'react';

type Props = {
  nodeId: string;
  fieldName: string;
};

const FieldResetToInitialLinearViewValueButton = ({ nodeId, fieldName }: Props) => {
  const { isValueChanged, resetToInitialLinearViewValue } = useFieldInitialLinearViewValue(nodeId, fieldName);

  return <FieldResetValueButton onClick={resetToInitialLinearViewValue} isDisabled={!isValueChanged} />;
};

export default memo(FieldResetToInitialLinearViewValueButton);
