import { Flex, useDisclosure } from '@chakra-ui/react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import IAICollapse from 'common/components/IAICollapse';
import ParamInfillMethod from './ParamInfillMethod';
import ParamInfillTilesize from './ParamInfillTilesize';
import ParamScaleBeforeProcessing from './ParamScaleBeforeProcessing';
import ParamScaledWidth from './ParamScaledWidth';
import ParamScaledHeight from './ParamScaledHeight';

const ParamInfillCollapse = () => {
  const { t } = useTranslation();
  const { isOpen, onToggle } = useDisclosure();

  return (
    <IAICollapse
      label={t('parameters.infillScalingHeader')}
      isOpen={isOpen}
      onToggle={onToggle}
    >
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamInfillMethod />
        <ParamInfillTilesize />
        <ParamScaleBeforeProcessing />
        <ParamScaledWidth />
        <ParamScaledHeight />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamInfillCollapse);
