/* eslint-disable i18next/no-literal-string */
import { Image } from '@chakra-ui/react';
import InvokeLogoYellow from 'assets/images/invoke-key-ylw-sm.svg';
import { InvText } from 'common/components/InvText/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { memo, useMemo, useRef } from 'react';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';
import { $customAppInfo } from '../../../app/store/nanostores/customAppInfo';
import { useStore } from '@nanostores/react';

const InvokeAILogoComponent = () => {
  const { data: appVersion } = useGetAppVersionQuery();
  const ref = useRef(null);
  const customAppInfo = useStore($customAppInfo);
  const tooltip = useMemo(() => {
    if (customAppInfo) {
      return <InvText fontWeight="semibold">{customAppInfo}</InvText>;
    }

    if (appVersion) {
      return <InvText fontWeight="semibold">v{appVersion.version}</InvText>;
    }
    return null;
  }, [appVersion, customAppInfo]);

  return (
    <InvTooltip placement="right" label={tooltip} p={1} px={2} gutter={16}>
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
    </InvTooltip>
  );
};

export default memo(InvokeAILogoComponent);
