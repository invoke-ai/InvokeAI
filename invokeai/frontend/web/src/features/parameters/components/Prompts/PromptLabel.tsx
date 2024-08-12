import { FormLabel } from '@invoke-ai/ui-library';

export const PromptLabel = ({ label }: { label: string }) => {
  return (
    <FormLabel color="base.450" fontSize="xs" pos="absolute" top={2} left={3}>
      {label}
    </FormLabel>
  );
};
