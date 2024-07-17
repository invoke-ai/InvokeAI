import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

type Props = {
  title: string;
  isSelected: boolean;
};

export const CanvasEntityTitle = memo(({ title, isSelected }: Props) => {
  return (
    <Text size="sm" fontWeight="semibold" userSelect="none" color={isSelected ? 'base.100' : 'base.300'}>
      {title}
    </Text>
  );
});

CanvasEntityTitle.displayName = 'CanvasEntityTitle';
