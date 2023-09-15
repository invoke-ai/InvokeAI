import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import InvokeAILogoImage from 'assets/images/logo.png';
import { useTranslation } from 'react-i18next';

export const ParamPositiveConditioningPopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.paramPositiveConditioning.paragraph')}
      heading={t('popovers.paramPositiveConditioning.heading')}
      buttonLabel="Learn more"
      buttonHref="http://google.com"
      image={InvokeAILogoImage}
      triggerComponent={props.children}
    />
  );
};
