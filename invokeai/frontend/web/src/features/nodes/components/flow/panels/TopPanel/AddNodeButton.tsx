import { IconButton } from '@invoke-ai/ui-library';
import { useAddNodeCmdk } from 'features/nodes/store/nodesSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

const AddNodeButton = () => {
  const addNodeCmdk = useAddNodeCmdk();
  const { t } = useTranslation();

  return (
    <IconButton
      tooltip={t('nodes.addNodeToolTip')}
      aria-label={t('nodes.addNode')}
      icon={<PiPlusBold />}
      onClick={addNodeCmdk.setTrue}
      pointerEvents="auto"
    />
  );
};

export default memo(AddNodeButton);
