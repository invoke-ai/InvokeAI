import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ControlNetControlModePopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.controlNetControlMode.paragraph')}
      heading={t('popovers.controlNetControlMode.heading')}
      triggerComponent={props.children}
    />
  );
};
