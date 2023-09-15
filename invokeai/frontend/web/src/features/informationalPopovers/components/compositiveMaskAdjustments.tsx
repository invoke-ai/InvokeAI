import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const CompositingMaskAdjustmentsPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.compositingMaskAdjustments.paragraph')}
      heading={t('popovers.compositingMaskAdjustments.heading')}
      triggerComponent={props.children}
    />
  );
};
