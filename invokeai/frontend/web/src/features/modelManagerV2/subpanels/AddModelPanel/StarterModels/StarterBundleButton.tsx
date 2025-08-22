import type { ButtonProps } from '@invoke-ai/ui-library';
import {
  Button,
  ConfirmationAlertDialog,
  Flex,
  ListItem,
  Text,
  UnorderedList,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useStarterBundleInstall } from 'features/modelManagerV2/hooks/useStarterBundleInstall';
import { useStarterBundleInstallStatus } from 'features/modelManagerV2/hooks/useStarterBundleInstallStatus';
import { t } from 'i18next';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import type { S } from 'services/api/types';

export const StarterBundleButton = ({ bundle, ...rest }: { bundle: S['StarterModelBundle'] } & ButtonProps) => {
  const { installBundle } = useStarterBundleInstall();
  const { install } = useStarterBundleInstallStatus(bundle);
  const { isOpen, onOpen, onClose } = useDisclosure();

  const onClickBundle = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      onOpen();
    },
    [onOpen]
  );
  const handleInstallBundle = useCallback(() => {
    installBundle(bundle);
  }, [installBundle, bundle]);

  return (
    <>
      <Button onClick={onClickBundle} isDisabled={install.length === 0} {...rest}>
        {bundle.name}
      </Button>
      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('modelManager.installBundle')}
        acceptCallback={handleInstallBundle}
        acceptButtonText={t('modelManager.install')}
        useInert={false}
      >
        <Flex rowGap={4} flexDirection="column">
          <Text fontWeight="bold">{t('modelManager.installBundleMsg1', { bundleName: bundle.name })}</Text>
          <Text>{t('modelManager.installBundleMsg2', { count: install.length })}</Text>
          <UnorderedList>
            {install.map((model, index) => (
              <ListItem key={index} wordBreak="break-all">
                <Text>{model.config.name}</Text>
              </ListItem>
            ))}
          </UnorderedList>
        </Flex>
      </ConfirmationAlertDialog>
    </>
  );
};
