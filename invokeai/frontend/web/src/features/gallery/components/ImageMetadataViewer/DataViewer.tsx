import type { FlexProps } from '@invoke-ai/ui-library';
import {
  Box,
  chakra,
  Flex,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Tooltip,
  useShiftModifier,
} from '@invoke-ai/ui-library';
import { getOverlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { useClipboard } from 'common/hooks/useClipboard';
import { isString } from 'es-toolkit/compat';
import { Formatter, TableCommaPlacement } from 'fracturedjsonjs';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { ChangeEvent, CSSProperties } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold, PiDownloadSimpleBold, PiXBold } from 'react-icons/pi';

const formatter = new Formatter();
formatter.Options.TableCommaPlacement = TableCommaPlacement.BeforePadding;
formatter.Options.OmitTrailingWhitespace = true;

type Props = {
  label: string;
  data: unknown;
  fileName?: string;
  withDownload?: boolean;
  withCopy?: boolean;
  extraCopyActions?: { label: string; getData: (data: unknown) => unknown }[];
  wrapData?: boolean;
  withSearch?: boolean;
} & FlexProps;

const overlayscrollbarsOptions = getOverlayScrollbarsParams({
  overflowX: 'scroll',
  overflowY: 'scroll',
}).options;

const ChakraPre = chakra('pre');

const DataViewer = (props: Props) => {
  const {
    label,
    data,
    fileName,
    withDownload = true,
    withCopy = true,
    extraCopyActions,
    wrapData = true,
    withSearch = false,
    ...rest
  } = props;
  const dataString = useMemo(() => (isString(data) ? data : formatter.Serialize(data)) ?? '', [data]);
  const shift = useShiftModifier();
  const clipboard = useClipboard();
  const [searchTerm, setSearchTerm] = useState('');
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

  const highlightedDataString = useMemo(() => {
    if (!searchTerm) {
      return dataString;
    }

    const regex = new RegExp(`(${escapeRegExp(searchTerm)})`, 'gi');
    const parts = dataString.split(regex);

    return parts.map((part, index) => {
      const isMatch = index % 2 === 1;
      if (!isMatch) {
        return <span key={index}>{part}</span>;
      }
      return (
        <chakra.mark key={index} bg="accent.700" color="accent.50" px={1} borderRadius="sm">
          {part}
        </chakra.mark>
      );
    });
  }, [dataString, searchTerm]);

  const handleChangeSearch = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  const handleClearSearch = useCallback(() => {
    setSearchTerm('');
  }, []);

  return (
    <Flex bg="base.800" borderRadius="base" flexGrow={1} w="full" h="full" position="relative" {...rest}>
      <Box position="absolute" top={0} left={0} right={0} bottom={0} overflow="auto" p={2} fontSize="sm">
        <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayscrollbarsOptions}>
          <ChakraPre whiteSpace={wrapData ? 'pre-wrap' : undefined}>{highlightedDataString}</ChakraPre>
        </OverlayScrollbarsComponent>
      </Box>
      <Flex position="absolute" top={0} insetInlineEnd={0} p={2} gap={2} alignItems="center">
        {withSearch && (
          <InputGroup size="sm" w={48}>
            <Input placeholder={t('common.search')} value={searchTerm} onChange={handleChangeSearch} />
            {searchTerm && (
              <InputRightElement h="full" pe={2}>
                <IconButton
                  aria-label={t('boards.clearSearch')}
                  icon={<PiXBold size={16} />}
                  variant="link"
                  opacity={0.7}
                  onClick={handleClearSearch}
                  size="sm"
                />
              </InputRightElement>
            )}
          </InputGroup>
        )}
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

const escapeRegExp = (value: string) => value.replace(/[-/\\^$*+?.()|[\]{}]/g, '$&');
