import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const DynamicPromptsTogglePopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.dynamicPromptsToggle.paragraph')}
      heading={t('popovers.dynamicPromptsToggle.heading')}
      triggerComponent={props.children}
    />
  );
};
