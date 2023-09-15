import { PropsWithChildren } from 'react';
import IAIInformationalPopover from '../../../common/components/IAIInformationalPopover';
import { useTranslation } from 'react-i18next';

export const ParamCFGScalePopover = (props: PropsWithChildren) => {
  const { t } = useTranslation();

  return (
    <IAIInformationalPopover
      paragraph={t('popovers.paramsCFGScale.paragraph')}
      heading={t('popovers.paramsCFGScale.heading')}
      triggerComponent={props.children}
    />
  );
};
