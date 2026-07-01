import { FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

interface Props {
  label: string;
  value: string | null | undefined;
}

export const ModelAttrView = memo(({ label, value }: Props) => {
  return (
    <FormControl flexDir="column" alignItems="flex-start" gap={0}>
      <FormLabel>{label}</FormLabel>
      <Text fontSize="md" noOfLines={1} wordBreak="break-all">
        {value || '-'}
      </Text>
    </FormControl>
  );
});

ModelAttrView.displayName = 'ModelAttrView';
