import { Badge, Button, Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold, PiPlusBold } from 'react-icons/pi';

type Props = {
  handleInstall: () => void;
  isInstalled: boolean;
};

export const ModelResultItemActions = memo(({ handleInstall, isInstalled }: Props) => {
  const { t } = useTranslation();

  return (
    <Flex gap={2} shrink={0} pt={1}>
      {isInstalled ? (
        // TODO: Add a link button to navigate to model
        <Badge
          variant="subtle"
          colorScheme="green"
          display="flex"
          gap={1}
          alignItems="center"
          borderRadius="base"
          h="24px"
        >
          <PiCheckBold size="14px" />
        </Badge>
      ) : (
        <Button
          onClick={handleInstall}
          rightIcon={<PiPlusBold size="14px" />}
          textTransform="uppercase"
          letterSpacing="wider"
          fontSize="9px"
          size="sm"
        >
          {t('modelManager.install')}
        </Button>
      )}
    </Flex>
  );
});

ModelResultItemActions.displayName = 'ModelResultItemActions';
