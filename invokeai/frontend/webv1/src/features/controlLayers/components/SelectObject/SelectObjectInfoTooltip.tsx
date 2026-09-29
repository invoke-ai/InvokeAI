import { Flex, Icon, ListItem, Text, Tooltip, UnorderedList } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';
import { Trans } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';

const Bold = (props: PropsWithChildren) => (
  <Text as="span" fontWeight="semibold">
    {props.children}
  </Text>
);

const components = { Bold: <Bold /> };

const SelectObjectHelpTooltipContent = memo(() => {
  return (
    <Flex gap={3} flexDir="column">
      <Text>
        <Trans i18nKey="controlLayers.selectObject.desc" components={components} />
      </Text>
      <UnorderedList>
        <ListItem>
          <Trans i18nKey="controlLayers.selectObject.visualMode1" components={components} />
        </ListItem>
        <ListItem>
          <Trans i18nKey="controlLayers.selectObject.visualMode2" components={components} />
        </ListItem>
        <ListItem>
          <Trans i18nKey="controlLayers.selectObject.visualMode3" components={components} />
        </ListItem>
      </UnorderedList>
      <Text>
        <Trans i18nKey="controlLayers.selectObject.promptModeDesc" components={components} />
      </Text>
      <UnorderedList>
        <ListItem>
          <Trans i18nKey="controlLayers.selectObject.promptMode1" components={components} />
        </ListItem>
        <ListItem>
          <Trans i18nKey="controlLayers.selectObject.promptMode2" components={components} />
        </ListItem>
      </UnorderedList>
    </Flex>
  );
});

SelectObjectHelpTooltipContent.displayName = 'SelectObjectHelpTooltipContent';

export const SelectObjectInfoTooltip = memo(() => {
  return (
    <Tooltip label={<SelectObjectHelpTooltipContent />} minW={420}>
      <Flex alignItems="center">
        <Icon as={PiInfoBold} color="base.500" />
      </Flex>
    </Tooltip>
  );
});

SelectObjectInfoTooltip.displayName = 'SelectObjectInfoTooltip';
