import { Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamCanvasRefineSteps from './ParamCanvasRefineSteps';
import ParamCanvasRefineStrength from './ParamCanvasRefineStrength';

const ParamCanvasRefinePassCollapse = () => {
  const { t } = useTranslation();

  return (
    <IAICollapse label={t('parameters.refinePassHeader')}>
      <Flex sx={{ flexDirection: 'column', gap: 2, paddingBottom: 2 }}>
        <ParamCanvasRefineSteps />
        <ParamCanvasRefineStrength />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamCanvasRefinePassCollapse);
