/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ModelConfig } from '@features/models/core/types';

import { Dialog, Input, Portal, Stack, Text } from '@chakra-ui/react';
import { modelPathSchema } from '@features/models/core/schemas';
import { updateModel } from '@features/models/data/api';
import { replaceModelInStore } from '@features/models/data/modelsStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button, CloseButton, Field } from '@platform/ui';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Update only the record's absolute path; never move files or repoint managed relative paths. */
export const UpdatePathDialog = ({
  model,
  onClose,
}: {
  model: Pick<ModelConfig, 'key' | 'name' | 'path'>;
  onClose: () => void;
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const [path, setPath] = useState(model.path);
  const [validationError, setValidationError] = useState<string | null>(null);
  const { isBusy, run } = useScopedAction();

  const handleSave = async () => {
    const parsed = modelPathSchema.safeParse(path);

    if (!parsed.success) {
      setValidationError(parsed.error.issues[0]?.message ?? t('models.invalidPathFormat'));

      return;
    }

    await run(
      async (owner) => {
        const updated = await updateModel(model.key, { path: parsed.data }, owner.signal);

        assertAccountScopeCurrent(owner);
        replaceModelInStore(updated);
        notify.success(t('models.pathUpdated'), model.name);
        onClose();
      },
      (message) => notify.error(t('models.pathUpdateFailed'), message)
    );
  };

  return (
    <Dialog.Root
      open
      size="md"
      onOpenChange={(event) => {
        if (!event.open) {
          onClose();
        }
      }}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
              <Stack gap="0.5">
                <Dialog.Title>{t('models.updatePath')}</Dialog.Title>
                <Dialog.Description>{t('models.updatePathDescription')}</Dialog.Description>
              </Stack>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="3">
                <Field label={t('models.currentPath')}>
                  <Text color="fg.muted" fontSize="2xs" overflowWrap="anywhere">
                    {model.path}
                  </Text>
                </Field>
                <Field error={validationError} label={t('models.newPath')}>
                  <Input
                    aria-invalid={validationError ? true : undefined}
                    placeholder={t('models.newPathPlaceholder')}
                    size="sm"
                    value={path}
                    onChange={(event) => {
                      setPath(event.currentTarget.value);
                      setValidationError(null);
                    }}
                  />
                </Field>
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button disabled={isBusy} size="xs" variant="ghost" onClick={onClose}>
                {t('common.cancel')}
              </Button>
              <Button
                disabled={path.trim() === model.path}
                loading={isBusy}
                size="xs"
                variant="solid"
                onClick={() => void handleSave()}
              >
                {t('models.updatePath')}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
