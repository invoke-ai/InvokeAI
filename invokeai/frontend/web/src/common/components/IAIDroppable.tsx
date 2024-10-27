import type { TypesafeDroppableData } from 'features/dnd/types';
import { memo } from 'react';

type IAIDroppableProps = {
  dropLabel?: string;
  disabled?: boolean;
  data?: TypesafeDroppableData;
};

const IAIDroppable = (props: IAIDroppableProps) => {
  return null;
};

export default memo(IAIDroppable);
