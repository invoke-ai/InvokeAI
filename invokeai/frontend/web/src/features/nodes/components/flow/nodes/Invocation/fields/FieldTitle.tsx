import { Flex, Text, forwardRef } from '@chakra-ui/react';
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
      sx={{
        position: 'relative',
        overflow: 'hidden',
        alignItems: 'center',
        justifyContent: 'flex-start',
        gap: 1,
        h: 'full',
        w: 'full',
      }}
    >
      <Text sx={{ fontWeight: isMissingInput ? 600 : 400 }}>
        {label || fieldTemplateTitle}
      </Text>
    </Flex>
  );
});

export default memo(FieldTitle);
