import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const NoiseEnablePopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.noiseEnable.paragraph')}
      heading={t('popovers.noiseEnable.heading')}
      triggerComponent={props.children}
    />
  );
};
