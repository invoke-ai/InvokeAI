import { Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamSeamBlur from './ParamSeamBlur';
import ParamSeamSize from './ParamSeamSize';
import ParamSeamSteps from './ParamSeamSteps';
import ParamSeamStrength from './ParamSeamStrength';
import ParamSeamThreshold from './ParamSeamThreshold';

const ParamSeamPaintingCollapse = () => {
  const { t } = useTranslation();

  return (
    <IAICollapse label={t('parameters.seamPaintingHeader')}>
      <Flex sx={{ flexDirection: 'column', gap: 2, paddingBottom: 2 }}>
        <ParamSeamSize />
        <ParamSeamBlur />
        <ParamSeamSteps />
        <ParamSeamStrength />
        <ParamSeamThreshold />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamSeamPaintingCollapse);
