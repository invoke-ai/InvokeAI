import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

type Props = {
  title: string;
};

export const EntityTitle = memo(({ title }: Props) => {
  return (
    <Text size="sm" fontWeight="semibold" userSelect="none" color="base.300">
      {title}
    </Text>
  );
});

EntityTitle.displayName = 'EntityTitle';
