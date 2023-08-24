import { Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamCanvasCoherenceSteps from './ParamCanvasCoherenceSteps';
import ParamCanvasCoherenceStrength from './ParamCanvasCoherenceStrength';

const ParamCanvasCoherencePassCollapse = () => {
  const { t } = useTranslation();

  return (
    <IAICollapse label={t('parameters.coherencePassHeader')}>
      <Flex sx={{ flexDirection: 'column', gap: 2, paddingBottom: 2 }}>
        <ParamCanvasCoherenceSteps />
        <ParamCanvasCoherenceStrength />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamCanvasCoherencePassCollapse);
