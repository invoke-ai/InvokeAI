/* eslint-disable i18next/no-literal-string */
import { Image, Text, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $logo } from 'app/store/nanostores/logo';
/** @knipignore */
import InvokeLogoYellow from 'public/assets/images/invoke-symbol-ylw-lrg.svg';
import { memo, useMemo, useRef } from 'react';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

const InvokeAILogoComponent = () => {
  const { data: appVersion } = useGetAppVersionQuery();
  const ref = useRef(null);
  const logoOverride = useStore($logo);
  const tooltip = useMemo(() => {
    if (appVersion) {
      return <Text fontWeight="semibold">v{appVersion.version}</Text>;
    }
    return null;
  }, [appVersion]);

  if (logoOverride) {
    return logoOverride;
  }

  return (
    <Tooltip placement="right" label={tooltip} p={1} px={2} gutter={16}>
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
    </Tooltip>
  );
};

export default memo(InvokeAILogoComponent);
