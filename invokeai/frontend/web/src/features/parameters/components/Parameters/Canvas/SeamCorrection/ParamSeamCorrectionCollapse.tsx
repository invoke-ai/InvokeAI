import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamSeamBlur from './ParamSeamBlur';
import ParamSeamSize from './ParamSeamSize';
import ParamSeamSteps from './ParamSeamSteps';
import ParamSeamStrength from './ParamSeamStrength';

const ParamSeamCorrectionCollapse = () => {
  const { t } = useTranslation();

  return (
    <IAICollapse label={t('parameters.seamCorrectionHeader')}>
      <ParamSeamSize />
      <ParamSeamBlur />
      <ParamSeamStrength />
      <ParamSeamSteps />
    </IAICollapse>
  );
};

export default memo(ParamSeamCorrectionCollapse);
