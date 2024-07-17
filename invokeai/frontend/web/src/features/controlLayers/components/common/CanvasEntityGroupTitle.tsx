import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

type Props = {
  title: string;
  isSelected: boolean;
};

export const CanvasEntityGroupTitle = memo(({ title, isSelected }: Props) => {
  return (
    <Text color={isSelected ? 'base.100' : 'base.300'} userSelect="none">
      {title}
    </Text>
  );
});

CanvasEntityGroupTitle.displayName = 'CanvasEntityGroupTitle';
