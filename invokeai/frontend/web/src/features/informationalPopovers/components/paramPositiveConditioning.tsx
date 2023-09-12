import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import InvokeAILogoImage from 'assets/images/logo.png';

export const ParamPositiveConditioningPopover = (props: PropsWithChildren) => {
  return (
    <IAIInformationalPopover
      heading="Prompt Box"
      paragraph="This is where you enter your prompt"
      buttonLabel="Learn more"
      buttonHref="http://google.com"
      image={InvokeAILogoImage}
      triggerComponent={props.children}
    />
  );
};
