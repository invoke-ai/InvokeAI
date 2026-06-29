import type { ComponentType } from 'react';

import { Card, Flex, Icon, Link, Stack, Text } from '@chakra-ui/react';
import { BookOpenTextIcon, ClapperboardIcon, CodeIcon, ExternalLinkIcon, MessagesSquareIcon } from 'lucide-react';

type ResourceLink = { href: string; icon: ComponentType; label: string };

const COMMUNITY: ResourceLink[] = [
  { href: 'https://discord.gg/ZmtBAhwWhy', icon: MessagesSquareIcon, label: 'Discord' },
  { href: 'https://github.com/invoke-ai/InvokeAI', icon: CodeIcon, label: 'GitHub' },
];

const GUIDES: ResourceLink[] = [
  { href: 'https://invoke-ai.github.io/InvokeAI/', icon: BookOpenTextIcon, label: 'Documentation' },
  { href: 'https://www.youtube.com/@invokeai', icon: ClapperboardIcon, label: 'YouTube' },
];

const RESOURCES = [
  {
    label: 'Community',
    resources: COMMUNITY,
  },
  {
    label: 'Guides',
    resources: GUIDES,
  },
];

const ResourceItem = ({ href, icon, label }: ResourceLink) => (
  <Link href={href} target="_blank" rel="noreferrer" asChild>
    <Flex align="center" gap={2} py={1}>
      <Icon as={icon} boxSize="3.5" />
      <Text fontSize="sm">{label}</Text>
      <Icon as={ExternalLinkIcon} boxSize="3" color="fg.subtle" ms="auto" />
    </Flex>
  </Link>
);

export const ResourceLinks = () => (
  <Stack>
    {RESOURCES.map((section) => (
      <Card.Root variant="outline" size="sm" key={section.label}>
        <Card.Header fontSize="sm">{section.label}</Card.Header>
        <Card.Body pt="2">
          {section.resources.map((resource) => (
            <ResourceItem {...resource} key={resource.href} />
          ))}
        </Card.Body>
      </Card.Root>
    ))}
  </Stack>
);
