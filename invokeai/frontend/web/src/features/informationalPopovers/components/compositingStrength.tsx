import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const CompositingStrengthPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.compositingStrength.paragraph')}
      heading={t('popovers.compositingStrength.heading')}
      triggerComponent={props.children}
    />
  );
};
