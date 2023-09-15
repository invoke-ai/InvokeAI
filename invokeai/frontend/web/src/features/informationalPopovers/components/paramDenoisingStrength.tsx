import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ParamDenoisingStrengthPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.paramsDenoisingStrength.paragraph')}
      heading={t('popovers.paramsDenoisingStrength.heading')}
      triggerComponent={props.children}
    />
  );
};
