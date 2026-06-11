import { HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { DownloadIcon, KeyRoundIcon } from 'lucide-react';

import { Button } from '../../components/ui/Button';
import { Field } from '../../components/ui/Field';
import { getHuggingFaceModels } from '../../models/api';
import { huggingFaceRepoSchema } from '../../models/schemas';
import { updateModelsUi, useModelsUi } from '../../models/uiStore';
import { useZodForm } from '../../models/useZodForm';
import { InstallSourceButton, SourceListItem } from './SourceListItem';
import { useInstallActions } from './useInstallActions';

/**
 * HuggingFace installs: diffusers repos install immediately; checkpoint repos
 * list their files (kept across tab switches) so the right one can be picked.
 */
export const HuggingFaceTab = () => {
  const { install, pendingSources } = useInstallActions();
  const { hfLookup } = useModelsUi();
  const form = useZodForm(huggingFaceRepoSchema, { repo: hfLookup?.repo ?? '' });

  const handleInstall = () =>
    form.handleSubmit(async ({ repo }) => {
      const lookup = await getHuggingFaceModels(repo);

      if (lookup.is_diffusers) {
        // A diffusers repo installs as one unit -- no file picking needed.
        updateModelsUi({ hfLookup: null });

        const ok = await install({ source: repo });

        if (ok) {
          form.reset();
        }

        return;
      }

      if (!lookup.urls || lookup.urls.length === 0) {
        throw new Error('No installable model files were found in this repo.');
      }

      updateModelsUi({ hfLookup: { repo, urls: lookup.urls } });
    });

  return (
    <Stack gap="4" maxW="40rem">
      <Field
        error={form.errors.repo ?? form.formError}
        helpText="Diffusers repos install immediately; checkpoint repos list their files below to choose from."
        label="HuggingFace Repo"
      >
        <HStack gap="1.5">
          <Input
            aria-invalid={form.errors.repo ? true : undefined}
            placeholder="owner/repo, e.g. black-forest-labs/FLUX.1-dev"
            size="sm"
            value={form.values.repo}
            onChange={(event) => form.setValue('repo', event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                void handleInstall();
              }
            }}
          />
          <Button
            disabled={form.values.repo.trim().length === 0}
            loading={form.isSubmitting}
            size="sm"
            variant="solid"
            onClick={() => void handleInstall()}
          >
            <Icon as={DownloadIcon} boxSize="3.5" />
            Install
          </Button>
        </HStack>
      </Field>

      {hfLookup ? (
        <Stack gap="1.5">
          <Text color="fg.muted" fontSize="2xs">
            {hfLookup.urls.length} file{hfLookup.urls.length === 1 ? '' : 's'} in {hfLookup.repo}:
          </Text>
          {hfLookup.urls.map((url) => {
            const fileName = url.split('/').at(-1) ?? url;

            return (
              <SourceListItem
                key={url}
                title={fileName}
                titleTooltip={url}
                trailing={
                  <InstallSourceButton
                    isPending={pendingSources.has(url)}
                    source={url}
                    onInstall={() => {
                      void install({ source: url });
                    }}
                  />
                }
              />
            );
          })}
        </Stack>
      ) : null}

      <HStack gap="1.5">
        <Icon as={KeyRoundIcon} boxSize="3.5" color="fg.subtle" />
        <Text color="fg.subtle" fontSize="2xs">
          Gated repos (like FLUX) need a HuggingFace token.
        </Text>
        <Button size="2xs" variant="ghost" onClick={() => updateModelsUi({ addTab: 'keys' })}>
          Manage API keys →
        </Button>
      </HStack>
    </Stack>
  );
};
