import { Badge } from '@invoke-ai/ui-library';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { memo } from 'react';
import type { BaseModelType } from 'services/api/types';

type Props = {
  base: BaseModelType;
};

const BASE_COLOR_MAP: Record<BaseModelType, string> = {
  any: 'base',
  'sd-1': 'green',
  'sd-2': 'teal',
  'sd-3': 'purple',
  sdxl: 'invokeBlue',
  'sdxl-refiner': 'invokeBlue',
  flux: 'gold',
  cogview4: 'orange',
};

const ModelBaseBadge = ({ base }: Props) => {
  return (
    <Badge flexGrow={0} colorScheme={BASE_COLOR_MAP[base]} variant="subtle" h="min-content">
      {MODEL_TYPE_SHORT_MAP[base]}
    </Badge>
  );
};

export default memo(ModelBaseBadge);
