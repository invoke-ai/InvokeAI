import { Flex, HStack, Text } from '@chakra-ui/react';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { Button } from '@platform/ui';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/** The empty center while its last view is a floating window: where the view went, and the two ways back. */
export const FloatedCenterView = ({
  dockLabel,
  instanceId,
  label,
  typeId,
}: {
  /** The Dock button's label, naming where the window returns to. */
  dockLabel: string;
  instanceId: string;
  label: string;
  typeId: string;
}) => {
  const { t } = useTranslation();
  const { widgets } = useWorkbenchCommands();
  const { focusFloating, focusRegion } = useWorkbenchFocus();
  const handleShow = useCallback(() => {
    widgets.revealFloating(instanceId);
    focusFloating(instanceId);
  }, [focusFloating, instanceId, widgets]);
  // Docking from here always gives the center its view back, so focus follows it.
  const handleDock = useCallback(() => {
    flushWorkbenchDrafts();
    widgets.dockFloating(instanceId);
    focusRegion('center', typeId);
  }, [focusRegion, instanceId, typeId, widgets]);

  return (
    <Flex align="center" direction="column" gap="3" h="full" justify="center" px="6" textAlign="center" w="full">
      <Text color="fg.muted" fontSize="sm">
        {t('widgets.floating.centerFloating', { label })}
      </Text>
      <HStack gap="2" wrap="wrap" justify="center">
        <Button size="xs" variant="outline" onClick={handleShow}>
          {t('widgets.floating.showWindow')}
        </Button>
        <Button size="xs" variant="outline" onClick={handleDock}>
          {dockLabel}
        </Button>
      </HStack>
    </Flex>
  );
};
