import { Button, ButtonGroup, Flex, Heading, Spacer } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const VectorLayerEditFooter = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const editSession = useStore(canvasManager.tool.tools.path.$editSession);

  if (!editSession) {
    return null;
  }

  return (
    <Flex
      bg="base.800"
      borderRadius="base"
      p={4}
      minW={420}
      flexDir="column"
      gap={4}
      shadow="dark-lg"
    >
      <Heading size="md" color="base.300" userSelect="none">
        {t('controlLayers.vectorEdit.title')}
      </Heading>
      <ButtonGroup isAttached={false} size="sm" w="full" alignItems="center">
        <Spacer />
        <Button onClick={canvasManager.tool.tools.path.acceptEditSession} variant="ghost">
          {t('common.accept')}
        </Button>
        <Button onClick={canvasManager.tool.tools.path.cancel} variant="ghost">
          {t('controlLayers.vectorEdit.discard')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

VectorLayerEditFooter.displayName = 'VectorLayerEditFooter';
