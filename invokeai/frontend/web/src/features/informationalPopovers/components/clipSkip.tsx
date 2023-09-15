import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ClipSkipPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.clipSkip.paragraph')}
      heading={t('popovers.clipSkip.heading')}
      triggerComponent={props.children}
    />
  );
};
