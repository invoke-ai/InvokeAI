import { Box, Flex, IconButton, Tooltip, useShiftModifier } from '@invoke-ai/ui-library';
import { getOverlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { useClipboard } from 'common/hooks/useClipboard';
import { Formatter } from 'fracturedjsonjs';
import { isString } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold, PiDownloadSimpleBold } from 'react-icons/pi';

const formatter = new Formatter();

type Props = {
  label: string;
  data: unknown;
  fileName?: string;
  withDownload?: boolean;
  withCopy?: boolean;
  extraCopyActions?: { label: string; getData: (data: unknown) => unknown }[];
};

const overlayscrollbarsOptions = getOverlayScrollbarsParams('scroll', 'scroll').options;

const DataViewer = (props: Props) => {
  const { label, data, fileName, withDownload = true, withCopy = true, extraCopyActions } = props;
  const dataString = useMemo(() => (isString(data) ? data : formatter.Serialize(data)) ?? '', [data]);
  const shift = useShiftModifier();
  const clipboard = useClipboard();
  const handleCopy = useCallback(() => {
    clipboard.writeText(dataString);
  }, [clipboard, dataString]);

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
    <Flex layerStyle="second" borderRadius="base" flexGrow={1} w="full" h="full" position="relative">
      <Box position="absolute" top={0} left={0} right={0} bottom={0} overflow="auto" p={4} fontSize="sm">
        <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayscrollbarsOptions}>
          <pre>{dataString}</pre>
        </OverlayScrollbarsComponent>
      </Box>
      <Flex position="absolute" top={0} insetInlineEnd={0} p={2}>
        {withDownload && (
          <Tooltip label={`${t('gallery.download')} ${label} JSON`}>
            <IconButton
              aria-label={`${t('gallery.download')} ${label} JSON`}
              icon={<PiDownloadSimpleBold size={16} />}
              variant="ghost"
              opacity={0.7}
              onClick={handleDownload}
            />
          </Tooltip>
        )}
        {withCopy && (
          <Tooltip label={`${t('gallery.copy')} ${label} JSON`}>
            <IconButton
              aria-label={`${t('gallery.copy')} ${label} JSON`}
              icon={<PiCopyBold size={16} />}
              variant="ghost"
              opacity={0.7}
              onClick={handleCopy}
            />
          </Tooltip>
        )}
        {shift &&
          extraCopyActions?.map(({ label, getData }) => (
            <ExtraCopyAction label={label} getData={getData} data={data} key={label} />
          ))}
      </Flex>
    </Flex>
  );
};

export default memo(DataViewer);

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

type ExtraCopyActionProps = {
  label: string;
  data: unknown;
  getData: (data: unknown) => unknown;
};
const ExtraCopyAction = ({ label, data, getData }: ExtraCopyActionProps) => {
  const { t } = useTranslation();
  const clipboard = useClipboard();
  const handleCopy = useCallback(() => {
    clipboard.writeText(JSON.stringify(getData(data), null, 2));
  }, [clipboard, data, getData]);

  return (
    <Tooltip label={`${t('gallery.copy')} ${label} JSON`}>
      <IconButton
        aria-label={`${t('gallery.copy')} ${label} JSON`}
        icon={<PiCopyBold size={16} />}
        variant="ghost"
        opacity={0.7}
        onClick={handleCopy}
      />
    </Tooltip>
  );
};
