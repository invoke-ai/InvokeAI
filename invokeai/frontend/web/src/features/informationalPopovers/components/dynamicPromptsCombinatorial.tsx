import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const DynamicPromptsCombinatorialPopover = (
  props: PropsWithChildren
) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.dynamicPromptsCombinatorial.paragraph')}
      heading={t('popovers.dynamicPromptsCombinatorial.heading')}
      triggerComponent={props.children}
    />
  );
};
