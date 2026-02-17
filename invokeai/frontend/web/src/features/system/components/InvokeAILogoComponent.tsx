import { Image, Text } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import InvokeLogoYellow from 'public/assets/images/invoke-symbol-ylw-lrg.svg';
import { memo, useMemo, useRef } from 'react';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

const InvokeAILogoComponent = () => {
  const { data: appVersion } = useGetAppVersionQuery();
  const ref = useRef(null);
  const tooltip = useMemo(() => {
    if (appVersion) {
      return <Text fontWeight="semibold">v{appVersion.version}</Text>;
    }
    return null;
  }, [appVersion]);

  return (
    <IAITooltip placement="right" label={tooltip} p={1} px={2} gutter={16}>
      <Image
        ref={ref}
        src={InvokeLogoYellow}
        alt="invoke-logo"
        w="24px"
        h="24px"
        minW="24px"
        minH="24px"
        userSelect="none"
      />
    </IAITooltip>
  );
};

export default memo(InvokeAILogoComponent);
