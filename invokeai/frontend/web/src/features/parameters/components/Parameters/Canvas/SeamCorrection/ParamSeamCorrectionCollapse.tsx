import ParamSeamBlur from './ParamSeamBlur';
import ParamSeamSize from './ParamSeamSize';
import ParamSeamSteps from './ParamSeamSteps';
import ParamSeamStrength from './ParamSeamStrength';
import { useDisclosure } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';

const ParamSeamCorrectionCollapse = () => {
  const { t } = useTranslation();
  const { isOpen, onToggle } = useDisclosure();

  return (
    <IAICollapse
      label={t('parameters.seamCorrectionHeader')}
      isOpen={isOpen}
      onToggle={onToggle}
    >
      <ParamSeamSize />
      <ParamSeamBlur />
      <ParamSeamStrength />
      <ParamSeamSteps />
    </IAICollapse>
  );
};

export default memo(ParamSeamCorrectionCollapse);
