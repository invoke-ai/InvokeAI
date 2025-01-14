import type { TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { useBatchGroupColorToken } from 'features/nodes/hooks/useBatchGroupColorToken';
import { memo } from 'react';

type Props = TextProps & {
  batchGroupId?: string;
};

export const BatchGroupId = memo(({ batchGroupId, ...rest }: Props) => {
  const batchGroupColorToken = useBatchGroupColorToken(batchGroupId);

  if (!batchGroupColorToken || !batchGroupId) {
    return null;
  }

  return (
    <Text fontWeight="semibold" color={batchGroupColorToken} {...rest}>
      {batchGroupId}
    </Text>
  );
});

BatchGroupId.displayName = 'BatchGroupId';
