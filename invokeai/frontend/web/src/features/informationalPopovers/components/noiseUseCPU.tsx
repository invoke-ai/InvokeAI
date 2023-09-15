import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const NoiseUseCPUPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.noiseUseCPU.paragraph')}
      heading={t('popovers.noiseUseCPU.heading')}
      triggerComponent={props.children}
    />
  );
};
