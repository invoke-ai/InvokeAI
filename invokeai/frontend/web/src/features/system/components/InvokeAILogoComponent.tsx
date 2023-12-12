/* eslint-disable i18next/no-literal-string */
import { Flex, Image, Text, Tooltip } from '@chakra-ui/react';
import InvokeAILogoImage from 'assets/images/logo.png';
import { memo, useRef } from 'react';
import { useHoverDirty } from 'react-use';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

interface Props {
  showVersion?: boolean;
}

const InvokeAILogoComponent = ({ showVersion = true }: Props) => {
  const { data: appVersion } = useGetAppVersionQuery();
  const ref = useRef(null);
  const isHovered = useHoverDirty(ref);

  return (
    <Flex alignItems="center" flexDirection="column" gap={5} ps={1} ref={ref}>
      <Tooltip
        sx={{
          background: 'base.100',
          _dark: {
            background: 'base.800',
          },
        }}
        placement="right"
        label={
          <Flex sx={{ gap: 1, alignItems: 'center' }}>
            <Text
              sx={{
                userSelect: 'none',
                color: 'base.800',
                _dark: { color: 'base.300' },
              }}
            >
              invoke <strong>ai</strong>
            </Text>
            {showVersion && isHovered && appVersion && (
              <Text
                sx={{
                  fontWeight: 600,
                  color: 'base.600',
                  fontSize: 12,
                  marginTop: 1,
                  _dark: {
                    color: 'base.400',
                  },
                }}
                variant="subtext"
              >
                {appVersion.version}
              </Text>
            )}
          </Flex>
        }
      >
        <Image
          src={InvokeAILogoImage}
          alt="invoke-ai-logo"
          sx={{
            w: '32px',
            h: '32px',
            minW: '32px',
            minH: '32px',
            userSelect: 'none',
          }}
        />
      </Tooltip>
      <Flex
        sx={{ gap: 3, alignItems: 'center', flexDirection: 'column' }}
      ></Flex>
    </Flex>
  );
};

export default memo(InvokeAILogoComponent);
