import { Text } from '@invoke-ai/ui-library';

export const PromptLabel = ({ label }: { label: string }) => {
  return (
    <Text variant="subtext" fontWeight="semibold" pos="absolute" top={1} left={2}>
      {label}
    </Text>
  );
};
