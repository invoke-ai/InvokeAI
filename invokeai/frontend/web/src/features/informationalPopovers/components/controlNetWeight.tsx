import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ControlNetWeightPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.controlNetWeight.paragraph')}
      heading={t('popovers.controlNetWeight.heading')}
      triggerComponent={props.children}
    />
  );
};
