import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ScaleBeforeProcessingPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.scaleBeforeProcessing.paragraph')}
      heading={t('popovers.scaleBeforeProcessing.heading')}
      triggerComponent={props.children}
    />
  );
};
