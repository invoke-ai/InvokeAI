import { HStack, Icon, Input, InputGroup, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui';
import { DownloadIcon, SearchIcon, XIcon } from 'lucide-react';
import { useCallback, type ChangeEvent, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

const SEARCH_ICON = <Icon as={SearchIcon} boxSize="3" color="fg.subtle" />;

/**
 * Shared header for an installable-source results panel (HuggingFace files,
 * folder-scan results): a summary line with an optional extra control and a
 * dismiss, over a filter box and an Install All action. Presentational only —
 * each consumer owns its own filter state, Install All predicate, and rows.
 */
export const ResultsListHeader = ({
  extra,
  installAllDisabled,
  installAllLabel,
  onClear,
  onInstallAll,
  onSearchChange,
  searchPlaceholder,
  searchValue,
  summary,
}: {
  extra?: ReactNode;
  installAllDisabled: boolean;
  installAllLabel?: string;
  onClear: () => void;
  onInstallAll: () => void;
  onSearchChange: (value: string) => void;
  searchPlaceholder?: string;
  searchValue: string;
  summary: ReactNode;
}) => {
  const { t } = useTranslation();
  const resolvedInstallAllLabel = installAllLabel ?? t('models.installAll');
  const resolvedSearchPlaceholder = searchPlaceholder ?? t('models.filterResults');
  const handleSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => onSearchChange(event.currentTarget.value),
    [onSearchChange]
  );

  return (
    <Stack gap="1.5">
      <HStack gap="2" justify="space-between" wrap="wrap">
        <Text color="fg.muted" fontSize="2xs">
          {summary}
        </Text>
        <HStack gap="3">
          {extra}
          <IconButton aria-label={t('models.dismissResults')} size="2xs" variant="ghost" onClick={onClear}>
            <Icon as={XIcon} boxSize="3" />
          </IconButton>
        </HStack>
      </HStack>
      <HStack gap="2">
        <InputGroup flex="1" startElement={SEARCH_ICON}>
          <Input
            aria-label={resolvedSearchPlaceholder}
            placeholder={resolvedSearchPlaceholder}
            size="xs"
            value={searchValue}
            onChange={handleSearchChange}
          />
        </InputGroup>
        <Button disabled={installAllDisabled} flexShrink={0} size="xs" variant="outline" onClick={onInstallAll}>
          <Icon as={DownloadIcon} boxSize="3" />
          {resolvedInstallAllLabel}
        </Button>
      </HStack>
    </Stack>
  );
};
