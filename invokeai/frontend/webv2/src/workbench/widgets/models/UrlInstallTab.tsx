import { Checkbox, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { Button, Field } from '@workbench/components/ui';
import { isCivitaiUrl } from '@workbench/models/apiKeys';
import { installSourceSchema } from '@workbench/models/schemas';
import { useZodForm } from '@workbench/models/useZodForm';
import { DownloadIcon } from 'lucide-react';

import { useInstallActions } from './useInstallActions';

/**
 * Install from a direct URL (Civitai, arbitrary hosts), a HuggingFace repo id,
 * or a path on the server's filesystem. Local paths can be installed in place
 * (left where they are) instead of copied into the models directory.
 */
export const UrlInstallTab = () => {
  const { install } = useInstallActions();
  const form = useZodForm(installSourceSchema, { accessToken: '', inplace: true, source: '' });

  const looksLocal = form.values.source.startsWith('/') || /^[A-Za-z]:[\\/]/.test(form.values.source);
  const looksCivitai = isCivitaiUrl(form.values.source);

  const handleInstall = () =>
    form.handleSubmit(async (values) => {
      const ok = await install({
        accessToken: values.accessToken === '' ? undefined : values.accessToken,
        inplace: looksLocal ? values.inplace : undefined,
        source: values.source,
      });

      if (ok) {
        form.reset();
      }
    });

  return (
    <Stack gap="4" maxW="40rem">
      <Field
        error={form.errors.source ?? form.formError}
        helpText="A model file URL, civitai.com model URL, HuggingFace repo id, or a path on the InvokeAI server."
        label="Model Source"
      >
        <Input
          aria-invalid={form.errors.source ? true : undefined}
          placeholder="https://civitai.com/… · owner/repo · /path/on/server/model.safetensors"
          size="sm"
          value={form.values.source}
          onChange={(event) => form.setValue('source', event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              void handleInstall();
            }
          }}
        />
      </Field>
      {looksLocal ? (
        <Checkbox.Root
          checked={form.values.inplace}
          colorPalette="accent"
          size="sm"
          onCheckedChange={(event) => form.setValue('inplace', event.checked === true)}
        >
          <Checkbox.HiddenInput />
          <Checkbox.Control />
          <Checkbox.Label fontSize="xs">Install in place (leave the file where it is)</Checkbox.Label>
        </Checkbox.Root>
      ) : null}
      {looksCivitai ? (
        <Text color="fg.subtle" fontSize="2xs">
          Your saved Civitai API key is attached automatically if this download requires login.
        </Text>
      ) : null}
      <Field
        error={form.errors.accessToken}
        helpText="Optional. Overrides saved keys for this one download."
        label="Access Token"
      >
        <Input
          placeholder="Only needed for protected URLs"
          size="sm"
          type="password"
          value={form.values.accessToken ?? ''}
          onChange={(event) => form.setValue('accessToken', event.currentTarget.value)}
        />
      </Field>
      <HStack>
        <Button
          disabled={form.values.source.trim().length === 0}
          loading={form.isSubmitting}
          size="sm"
          variant="solid"
          onClick={() => void handleInstall()}
        >
          <Icon as={DownloadIcon} boxSize="3.5" />
          Install Model
        </Button>
      </HStack>
      <Text color="fg.subtle" fontSize="2xs">
        Manage HuggingFace, Civitai, and external provider keys in the API Keys tab.
      </Text>
    </Stack>
  );
};
