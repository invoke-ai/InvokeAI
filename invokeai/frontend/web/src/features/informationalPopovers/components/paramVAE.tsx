import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ParamVAEPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.paramVAE.paragraph')}
      heading={t('popovers.paramVAE.heading')}
      triggerComponent={props.children}
    />
  );
};
