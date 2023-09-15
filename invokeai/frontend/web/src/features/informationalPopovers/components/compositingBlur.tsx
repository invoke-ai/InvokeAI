import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const CompositingBlurPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.compositingBlur.paragraph')}
      heading={t('popovers.compositingBlur.heading')}
      triggerComponent={props.children}
    />
  );
};
