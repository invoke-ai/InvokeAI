import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

import { usePromptTokenCount } from './usePromptTokenCount';

type PromptTokenCounterProps = {
  promptText: string;
};

export const PromptTokenCounter = memo(({ promptText }: PromptTokenCounterProps) => {
  const tokenState = usePromptTokenCount(promptText);

  if (!tokenState) {
    return null;
  }

  const { count, limit, isNearLimit, isOverLimit } = tokenState;

  let color = 'base.400';
  if (isOverLimit) {
    color = 'error.400';
  } else if (isNearLimit) {
    color = 'warning.400';
  }

  return (
    <Text
      variant="subtext"
      fontWeight="semibold"
      fontSize="xs"
      pos="absolute"
      top={1}
      right={12}
      color={color}
      pointerEvents="none"
      userSelect="none"
    >
      Tokens: {count} / {limit}
    </Text>
  );
});

PromptTokenCounter.displayName = 'PromptTokenCounter';
