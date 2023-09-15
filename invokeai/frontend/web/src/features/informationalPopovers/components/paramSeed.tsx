import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ParamSeedPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.paramSeed.paragraph')}
      heading={t('popovers.paramSeed.heading')}
      triggerComponent={props.children}
    />
  );
};
