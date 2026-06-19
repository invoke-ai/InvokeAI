import type { ReactNode } from 'react';

import { HStack, Icon, Input, InputGroup, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@workbench/components/ui';
import { DownloadIcon, SearchIcon, XIcon } from 'lucide-react';

/**
 * Shared header for an installable-source results panel (HuggingFace files,
 * folder-scan results): a summary line with an optional extra control and a
 * dismiss, over a filter box and an Install All action. Presentational only —
 * each consumer owns its own filter state, Install All predicate, and rows.
 */
export const ResultsListHeader = ({
  extra,
  installAllDisabled,
  installAllLabel = 'Install all',
  onClear,
  onInstallAll,
  onSearchChange,
  searchPlaceholder = 'Filter results',
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
}) => (
  <Stack gap="1.5">
    <HStack gap="2" justify="space-between" wrap="wrap">
      <Text color="fg.muted" fontSize="2xs">
        {summary}
      </Text>
      <HStack gap="3">
        {extra}
        <IconButton aria-label="Dismiss results" size="2xs" variant="ghost" onClick={onClear}>
          <Icon as={XIcon} boxSize="3" />
        </IconButton>
      </HStack>
    </HStack>
    <HStack gap="2">
      <InputGroup flex="1" startElement={<Icon as={SearchIcon} boxSize="3" color="fg.subtle" />}>
        <Input
          aria-label={searchPlaceholder}
          placeholder={searchPlaceholder}
          size="xs"
          value={searchValue}
          onChange={(event) => onSearchChange(event.currentTarget.value)}
        />
      </InputGroup>
      <Button disabled={installAllDisabled} flexShrink={0} size="xs" variant="outline" onClick={onInstallAll}>
        <Icon as={DownloadIcon} boxSize="3" />
        {installAllLabel}
      </Button>
    </HStack>
  </Stack>
);
