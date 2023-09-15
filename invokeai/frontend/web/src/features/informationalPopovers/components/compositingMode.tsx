import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const CompositingModePopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.compositingMode.paragraph')}
      heading={t('popovers.compositingMode.heading')}
      triggerComponent={props.children}
    />
  );
};
