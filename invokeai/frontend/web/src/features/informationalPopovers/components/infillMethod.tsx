import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const InfillMethodPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.infillMethod.paragraph')}
      heading={t('popovers.infillMethod.heading')}
      triggerComponent={props.children}
    />
  );
};
