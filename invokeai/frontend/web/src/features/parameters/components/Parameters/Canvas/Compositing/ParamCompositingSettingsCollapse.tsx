import { Divider, Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import SubParametersWrapper from '../../SubParametersWrapper';
import ParamCanvasCoherenceMode from './CoherencePass/ParamCanvasCoherenceMode';
import ParamCanvasCoherenceSteps from './CoherencePass/ParamCanvasCoherenceSteps';
import ParamCanvasCoherenceStrength from './CoherencePass/ParamCanvasCoherenceStrength';
import ParamMaskBlur from './MaskAdjustment/ParamMaskBlur';
import ParamMaskBlurMethod from './MaskAdjustment/ParamMaskBlurMethod';

const ParamCompositingSettingsCollapse = () => {
  const { t } = useTranslation();

  return (
    <IAICollapse label={t('parameters.compositingSettingsHeader')}>
      <Flex sx={{ flexDirection: 'column', gap: 2 }}>
        <SubParametersWrapper label={t('parameters.coherencePassHeader')}>
          <ParamCanvasCoherenceMode />
          <ParamCanvasCoherenceSteps />
          <ParamCanvasCoherenceStrength />
        </SubParametersWrapper>
        <Divider />
        <SubParametersWrapper label={t('parameters.maskAdjustmentsHeader')}>
          <ParamMaskBlur />
          <ParamMaskBlurMethod />
        </SubParametersWrapper>
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamCompositingSettingsCollapse);
