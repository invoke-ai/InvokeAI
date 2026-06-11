import { Card, Flex, Icon, Link, Stack, Text } from '@chakra-ui/react';
import { BookOpenTextIcon, ClapperboardIcon, CodeIcon, ExternalLinkIcon, MessagesSquareIcon } from 'lucide-react';
import type { ComponentType } from 'react';

/**
 * Home's side rail: pointers out to the docs and community, plus the
 * placeholder slot where in-app tutorials will land.
 */

const RESOURCES: { href: string; icon: ComponentType; label: string }[] = [
  { href: 'https://invoke-ai.github.io/InvokeAI/', icon: BookOpenTextIcon, label: 'Documentation' },
  { href: 'https://www.youtube.com/@invokeai', icon: ClapperboardIcon, label: 'YouTube' },
  { href: 'https://discord.gg/ZmtBAhwWhy', icon: MessagesSquareIcon, label: 'Discord' },
  { href: 'https://github.com/invoke-ai/InvokeAI', icon: CodeIcon, label: 'GitHub' },
];

export const ResourceLinks = () => (
  <Stack>
    <Card.Root variant="outline">
      <Card.Header>Resources</Card.Header>
      <Card.Body>
        {RESOURCES.map((resource) => (
          <Link href={resource.href} target="_blank" rel="noreferrer" asChild key={resource.href}>
            <Flex align="center" gap={2} py={1}>
              <Icon as={resource.icon} boxSize="3.5" />
              <Text fontSize="sm">{resource.label}</Text>
              <Icon as={ExternalLinkIcon} boxSize="3" color="fg.subtle" ms="auto" />
            </Flex>
          </Link>
        ))}
      </Card.Body>
    </Card.Root>

    {/* TODO: Tutorials */}
  </Stack>
);
