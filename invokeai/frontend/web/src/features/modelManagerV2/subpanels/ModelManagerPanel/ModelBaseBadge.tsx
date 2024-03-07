import { Badge } from '@invoke-ai/ui-library';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { memo } from 'react';
import type { BaseModelType } from 'services/api/types';

type Props = {
  base: BaseModelType;
};

const ModelBaseBadge = ({ base }: Props) => {
  return (
    <Badge flexGrow={0} colorScheme="invokeBlue" variant="subtle">
      {MODEL_TYPE_SHORT_MAP[base]}
    </Badge>
  );
};

export default memo(ModelBaseBadge);
