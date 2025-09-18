import { Badge } from '@invoke-ai/ui-library';
import { MODEL_BASE_TO_COLOR, MODEL_BASE_TO_SHORT_NAME } from 'features/modelManagerV2/models';
import { memo } from 'react';
import type { BaseModelType } from 'services/api/types';

type Props = {
  base: BaseModelType;
};

const ModelBaseBadge = ({ base }: Props) => {
  return (
    <Badge flexGrow={0} flexShrink={0} colorScheme={MODEL_BASE_TO_COLOR[base]} variant="subtle" h="min-content">
      {MODEL_BASE_TO_SHORT_NAME[base]}
    </Badge>
  );
};

export default memo(ModelBaseBadge);
