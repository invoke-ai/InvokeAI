/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Icon, Input, Popover, Portal, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@workbench/components/ui';
import { KeyRoundIcon } from 'lucide-react';

/** Key icon -> popover with a one-off access token and a jump to the keys tab. */
export const AccessTokenPopover = ({
  onChange,
  onManageKeys,
  value,
}: {
  onChange: (value: string) => void;
  onManageKeys: () => void;
  value: string;
}) => (
  <Popover.Root lazyMount positioning={{ placement: 'bottom-end' }}>
    <Popover.Trigger asChild>
      <IconButton
        aria-label="Access token for this download"
        color={value.trim() === '' ? 'fg.muted' : 'accent.solid'}
        size="sm"
        variant="outline"
      >
        <Icon as={KeyRoundIcon} boxSize="4" />
      </IconButton>
    </Popover.Trigger>
    <Portal>
      <Popover.Positioner>
        <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="20rem">
          <Popover.Body p="2.5">
            <Stack gap="2">
              <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
                Access Token
              </Text>
              <Input
                aria-label="Access token"
                placeholder="Only needed for protected sources"
                size="xs"
                type="password"
                value={value}
                onChange={(event) => onChange(event.currentTarget.value)}
              />
              <Text color="fg.subtle" fontSize="2xs">
                Attached to the next Pull only. For saved HuggingFace, Civitai, and provider keys, use the API Keys tab.
              </Text>
              <Button alignSelf="start" size="2xs" variant="ghost" onClick={onManageKeys}>
                Manage API keys {'->'}
              </Button>
            </Stack>
          </Popover.Body>
        </Popover.Content>
      </Popover.Positioner>
    </Portal>
  </Popover.Root>
);
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
