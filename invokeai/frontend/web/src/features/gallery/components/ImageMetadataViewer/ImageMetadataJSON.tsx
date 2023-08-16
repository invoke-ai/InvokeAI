import { Box, Flex, IconButton, Tooltip } from '@chakra-ui/react';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { useCallback, useMemo } from 'react';
import { FaCopy, FaSave } from 'react-icons/fa';

type Props = {
  label: string;
  jsonObject: object;
  fileName?: string;
};

const ImageMetadataJSON = (props: Props) => {
  const { label, jsonObject, fileName } = props;
  const jsonString = useMemo(
    () => JSON.stringify(jsonObject, null, 2),
    [jsonObject]
  );

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(jsonString);
  }, [jsonString]);

  const handleSave = useCallback(() => {
    const blob = new Blob([jsonString]);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${fileName || label}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }, [jsonString, label, fileName]);

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
          <pre>{jsonString}</pre>
        </OverlayScrollbarsComponent>
      </Box>
      <Flex sx={{ position: 'absolute', top: 0, insetInlineEnd: 0, p: 2 }}>
        <Tooltip label={`Save ${label} JSON`}>
          <IconButton
            aria-label={`Save ${label} JSON`}
            icon={<FaSave />}
            variant="ghost"
            opacity={0.7}
            onClick={handleSave}
          />
        </Tooltip>
        <Tooltip label={`Copy ${label} JSON`}>
          <IconButton
            aria-label={`Copy ${label} JSON`}
            icon={<FaCopy />}
            variant="ghost"
            opacity={0.7}
            onClick={handleCopy}
          />
        </Tooltip>
      </Flex>
    </Flex>
  );
};

export default ImageMetadataJSON;
