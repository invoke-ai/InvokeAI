import { Box, Flex } from '@chakra-ui/react';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { isString } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
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

  const { t } = useTranslation();

  return (
    <Flex
      layerStyle="second"
      borderRadius="base"
      flexGrow={1}
      w="full"
      h="full"
      position="relative"
    >
      <Box
        position="absolute"
        top={0}
        left={0}
        right={0}
        bottom={0}
        overflow="auto"
        p={4}
        fontSize="sm"
      >
        <OverlayScrollbarsComponent
          defer
          style={overlayScrollbarsStyles}
          options={overlayScrollbarsParams.options}
        >
          <pre>{dataString}</pre>
        </OverlayScrollbarsComponent>
      </Box>
      <Flex position="absolute" top={0} insetInlineEnd={0} p={2}>
        {withDownload && (
          <InvTooltip label={`${t('gallery.download')} ${label} JSON`}>
            <InvIconButton
              aria-label={`${t('gallery.download')} ${label} JSON`}
              icon={<FaDownload />}
              variant="ghost"
              opacity={0.7}
              onClick={handleDownload}
            />
          </InvTooltip>
        )}
        {withCopy && (
          <InvTooltip label={`${t('gallery.copy')} ${label} JSON`}>
            <InvIconButton
              aria-label={`${t('gallery.copy')} ${label} JSON`}
              icon={<FaCopy />}
              variant="ghost"
              opacity={0.7}
              onClick={handleCopy}
            />
          </InvTooltip>
        )}
      </Flex>
    </Flex>
  );
};

export default memo(DataViewer);

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};
