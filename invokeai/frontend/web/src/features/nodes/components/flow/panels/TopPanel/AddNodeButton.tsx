import { IconButton } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import { $addNodeCmdk } from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

const AddNodeButton = () => {
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    $addNodeCmdk.set(true);
  }, []);

  return (
    <IAITooltip label={t('nodes.addNodeToolTip')}>
      <IconButton aria-label={t('nodes.addNode')} icon={<PiPlusBold />} onClick={onClick} pointerEvents="auto" />
    </IAITooltip>
  );
};

export default memo(AddNodeButton);
