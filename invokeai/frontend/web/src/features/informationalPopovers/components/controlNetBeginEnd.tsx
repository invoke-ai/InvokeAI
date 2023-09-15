import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ControlNetBeginEndPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.controlNetBeginEnd.paragraph')}
      heading={t('popovers.controlNetBeginEnd.heading')}
      triggerComponent={props.children}
    />
  );
};
