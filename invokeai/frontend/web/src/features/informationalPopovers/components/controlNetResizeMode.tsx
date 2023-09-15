import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ControlNetResizeModePopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.controlNetResizeMode.paragraph')}
      heading={t('popovers.controlNetResizeMode.heading')}
      triggerComponent={props.children}
    />
  );
};
