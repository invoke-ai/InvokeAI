import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const LoraPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.lora.paragraph')}
      heading={t('popovers.lora.heading')}
      triggerComponent={props.children}
    />
  );
};
