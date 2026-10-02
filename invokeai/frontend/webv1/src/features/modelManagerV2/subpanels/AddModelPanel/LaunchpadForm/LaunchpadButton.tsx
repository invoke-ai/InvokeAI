import { Button, Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { IconType } from 'react-icons';

export const LaunchpadButton = memo(
  (props: { onClick: () => void; icon: IconType; title: string; description: string }) => {
    const { onClick, icon, title, description } = props;

    return (
      <Button onClick={onClick} variant="outline" p={4} textAlign="left" flexDir="column" gap={2} h="unset">
        <Flex alignItems="center" gap={4} w="full">
          <Icon as={icon} boxSize={8} color="base.300" />
          <Heading size="sm" color="base.100">
            {title}
          </Heading>
        </Flex>
        <Text lineHeight="1.4" flex="1" whiteSpace="normal" wordBreak="break-word">
          {description}
        </Text>
      </Button>
    );
  }
);

LaunchpadButton.displayName = 'LaunchpadButton';
