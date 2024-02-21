import { FormControl, FormLabel, Text } from '@invoke-ai/ui-library';

interface Props {
  label: string;
  value: string | null | undefined;
}

export const ModelAttrView = ({ label, value }: Props) => {
  return (
    <FormControl flexDir="column" alignItems="flex-start" gap={0}>
      <FormLabel>{label}</FormLabel>
      <Text fontSize="md">{value || '-'}</Text>
    </FormControl>
  );
};
