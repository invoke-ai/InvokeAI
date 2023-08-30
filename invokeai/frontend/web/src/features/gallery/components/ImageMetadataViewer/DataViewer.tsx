import { Box, Flex, IconButton, Tooltip } from '@chakra-ui/react';
import { isString } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useCallback, useMemo } from 'react';
import { FaCopy, FaDownload } from 'react-icons/fa';

type Props = {
  label: string;
  data: object | string;
  fileName?: string;
  withDownload?: boolean;
  withCopy?: boolean;
};

const DataViewer = (props: Props) => {
  const { label, data, fileName, withDownload = true, withCopy = true } = props;
  const dataString = useMemo(
    () => (isString(data) ? data : JSON.stringify(data, null, 2)),
    [data]
  );

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(dataString);
  }, [dataString]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([dataString]);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${fileName || label}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }, [dataString, label, fileName]);

  return (
    <Flex
      layerStyle="second"
      sx={{
        borderRadius: 'base',
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
          fontSize: 'sm',
        }}
      >
        <OverlayScrollbarsComponent
          defer
          style={{ height: '100%', width: '100%' }}
          options={{
            scrollbars: {
              visibility: 'auto',
              autoHide: 'scroll',
              autoHideDelay: 1300,
              theme: 'os-theme-dark',
            },
          }}
        >
          <pre>{dataString}</pre>
        </OverlayScrollbarsComponent>
      </Box>
      <Flex sx={{ position: 'absolute', top: 0, insetInlineEnd: 0, p: 2 }}>
        {withDownload && (
          <Tooltip label={`Download ${label} JSON`}>
            <IconButton
              aria-label={`Download ${label} JSON`}
              icon={<FaDownload />}
              variant="ghost"
              opacity={0.7}
              onClick={handleDownload}
            />
          </Tooltip>
        )}
        {withCopy && (
          <Tooltip label={`Copy ${label} JSON`}>
            <IconButton
              aria-label={`Copy ${label} JSON`}
              icon={<FaCopy />}
              variant="ghost"
              opacity={0.7}
              onClick={handleCopy}
            />
          </Tooltip>
        )}
      </Flex>
    </Flex>
  );
};

export default memo(DataViewer);
