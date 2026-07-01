import { IconButton } from '@invoke-ai/ui-library';
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
    <IconButton
      tooltip={t('nodes.addNodeToolTip')}
      aria-label={t('nodes.addNode')}
      icon={<PiPlusBold />}
      onClick={onClick}
      pointerEvents="auto"
    />
  );
};

export default memo(AddNodeButton);
