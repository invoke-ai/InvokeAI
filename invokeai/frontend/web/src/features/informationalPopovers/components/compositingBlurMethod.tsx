import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const CompositingBlurMethodPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.compositingBlurMethod.paragraph')}
      heading={t('popovers.compositingBlurMethod.heading')}
      triggerComponent={props.children}
    />
  );
};
