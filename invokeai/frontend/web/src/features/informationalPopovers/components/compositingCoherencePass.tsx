import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const CompositingCoherencePassPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.compositingCoherencePass.paragraph')}
      heading={t('popovers.compositingCoherencePass.heading')}
      triggerComponent={props.children}
    />
  );
};
