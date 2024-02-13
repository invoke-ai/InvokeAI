import { Flex, forwardRef, Text } from '@invoke-ai/ui-library';
import { useFieldLabel } from 'features/nodes/hooks/useFieldLabel';
import { useFieldTemplateTitle } from 'features/nodes/hooks/useFieldTemplateTitle';
import { memo } from 'react';

interface Props {
  nodeId: string;
  fieldName: string;
  kind: 'input' | 'output';
  isMissingInput?: boolean;
}

const FieldTitle = forwardRef((props: Props, ref) => {
  const { nodeId, fieldName, kind, isMissingInput = false } = props;
  const label = useFieldLabel(nodeId, fieldName);
  const fieldTemplateTitle = useFieldTemplateTitle(nodeId, fieldName, kind);

  return (
    <Flex
      ref={ref}
      position="relative"
      overflow="hidden"
      alignItems="center"
      justifyContent="flex-start"
      gap={1}
      h="full"
      w="full"
    >
      <Text fontWeight={isMissingInput ? 'bold' : 'normal'}>{label || fieldTemplateTitle}</Text>
    </Flex>
  );
});

export default memo(FieldTitle);
