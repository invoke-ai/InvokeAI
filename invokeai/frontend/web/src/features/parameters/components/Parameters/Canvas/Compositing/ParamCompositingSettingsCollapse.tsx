import { Divider, Flex, Text } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { PropsWithChildren, memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamCanvasCoherenceSteps from './CoherencePass/ParamCanvasCoherenceSteps';
import ParamCanvasCoherenceStrength from './CoherencePass/ParamCanvasCoherenceStrength';
import ParamMaskBlur from './MaskAdjustment/ParamMaskBlur';
import ParamMaskBlurMethod from './MaskAdjustment/ParamMaskBlurMethod';

const ParamCompositingSettingsCollapse = () => {
  const { t } = useTranslation();

  return (
    <IAICollapse label={t('parameters.compositingSettingsHeader')}>
      <Flex sx={{ flexDirection: 'column', gap: 2 }}>
        <CompositingSettingsWrapper>
          <Text fontSize="sm" fontWeight="bold">
            {t('parameters.maskAdjustmentsHeader')}
          </Text>
          <ParamMaskBlur />
          <ParamMaskBlurMethod />
        </CompositingSettingsWrapper>
        <Divider />
        <CompositingSettingsWrapper>
          <Text fontSize="sm" fontWeight="bold">
            {t('parameters.coherencePassHeader')}
          </Text>
          <ParamCanvasCoherenceSteps />
          <ParamCanvasCoherenceStrength />
        </CompositingSettingsWrapper>
      </Flex>
    </IAICollapse>
  );
};

const CompositingSettingsWrapper = memo((props: PropsWithChildren) => (
  <Flex
    sx={{
      flexDir: 'column',
      gap: 2,
      bg: 'base.100',
      px: 4,
      pt: 2,
      pb: 4,
      borderRadius: 'base',
      _dark: {
        bg: 'base.750',
      },
    }}
  >
    {props.children}
  </Flex>
));

CompositingSettingsWrapper.displayName = 'CompositingSettingsWrapper';

export default memo(ParamCompositingSettingsCollapse);
