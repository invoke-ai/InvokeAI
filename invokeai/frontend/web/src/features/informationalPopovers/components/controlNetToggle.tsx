import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ControlNetTogglePopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.controlNetToggle.paragraph')}
      heading={t('popovers.controlNetToggle.heading')}
      triggerComponent={props.children}
    />
  );
};
