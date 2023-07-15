import { Box, Flex, IconButton, Tooltip } from '@chakra-ui/react';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { useMemo } from 'react';
import { FaCopy } from 'react-icons/fa';

type Props = {
  copyTooltip: string;
  jsonObject: object;
};

const ImageMetadataJSON = (props: Props) => {
  const { copyTooltip, jsonObject } = props;
  const jsonString = useMemo(
    () => JSON.stringify(jsonObject, null, 2),
    [jsonObject]
  );

  return (
    <Flex
      sx={{
        borderRadius: 'base',
        bg: 'whiteAlpha.500',
        _dark: { bg: 'blackAlpha.500' },
        flexGrow: 1,
        w: 'full',
        h: 'full',
        position: 'relative',
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          overflow: 'auto',
          p: 4,
        }}
      >
        <OverlayScrollbarsComponent
          defer
          style={{ height: '100%', width: '100%' }}
          options={{
            scrollbars: {
              visibility: 'auto',
              autoHide: 'move',
              autoHideDelay: 1300,
              theme: 'os-theme-dark',
            },
          }}
        >
          <pre>{jsonString}</pre>
        </OverlayScrollbarsComponent>
      </Box>
      <Flex sx={{ position: 'absolute', top: 0, insetInlineEnd: 0, p: 2 }}>
        <Tooltip label={copyTooltip}>
          <IconButton
            aria-label={copyTooltip}
            icon={<FaCopy />}
            variant="ghost"
            onClick={() => navigator.clipboard.writeText(jsonString)}
          />
        </Tooltip>
      </Flex>
    </Flex>
  );
};

export default ImageMetadataJSON;
