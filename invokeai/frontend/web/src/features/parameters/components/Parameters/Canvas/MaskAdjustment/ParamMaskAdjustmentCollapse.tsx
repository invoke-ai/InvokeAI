import { Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamMaskBlur from './ParamMaskBlur';
import ParamMaskBlurMethod from './ParamMaskBlurMethod';

const ParamMaskAdjustmentCollapse = () => {
  const { t } = useTranslation();

  return (
    <IAICollapse label={t('parameters.maskAdjustmentsHeader')}>
      <Flex sx={{ flexDirection: 'column', gap: 2 }}>
        <ParamMaskBlur />
        <ParamMaskBlurMethod />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamMaskAdjustmentCollapse);
