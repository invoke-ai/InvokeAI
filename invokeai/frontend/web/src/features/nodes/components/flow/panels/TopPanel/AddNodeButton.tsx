import { IconButton } from '@invoke-ai/ui-library';
import { openAddNodePopover } from 'features/nodes/store/nodesSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

const AddNodeButton = () => {
  const { t } = useTranslation();

  return (
    <IconButton
      tooltip={t('nodes.addNodeToolTip')}
      aria-label={t('nodes.addNode')}
      icon={<PiPlusBold />}
      onClick={openAddNodePopover}
      pointerEvents="auto"
    />
  );
};

export default memo(AddNodeButton);
