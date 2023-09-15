import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ParamVAEPrecisionPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.paramVAEPrecision.paragraph')}
      heading={t('popovers.paramVAEPrecision.heading')}
      triggerComponent={props.children}
    />
  );
};
