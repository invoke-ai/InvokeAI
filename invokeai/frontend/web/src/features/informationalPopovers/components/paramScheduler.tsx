import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ParamSchedulerPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.paramScheduler.paragraph')}
      heading={t('popovers.paramScheduler.heading')}
      triggerComponent={props.children}
    />
  );
};
