import { Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamRefineSteps from './ParamRefineSteps';
import ParamRefineStrength from './ParamRefineStrength';

const ParamRefinePassCollapse = () => {
  const { t } = useTranslation();

  return (
    <IAICollapse label={t('parameters.refinePassHeader')}>
      <Flex sx={{ flexDirection: 'column', gap: 2, paddingBottom: 2 }}>
        <ParamRefineSteps />
        <ParamRefineStrength />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamRefinePassCollapse);
