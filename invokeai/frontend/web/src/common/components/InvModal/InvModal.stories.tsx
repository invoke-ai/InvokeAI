import { useDisclosure } from '@chakra-ui/react';
import type { Meta, StoryObj } from '@storybook/react';
import { InvButton } from 'common/components/InvButton/InvButton';

import {
  InvModal,
  InvModalBody,
  InvModalCloseButton,
  InvModalContent,
  InvModalFooter,
  InvModalHeader,
  InvModalOverlay,
} from './wrapper';

const meta: Meta<typeof InvModal> = {
  title: 'Primitives/InvModal',
  tags: ['autodocs'],
  component: InvModal,
  args: {
    colorScheme: 'base',
  },
};

export default meta;
type Story = StoryObj<typeof InvModal>;

const Component = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  return (
    <>
      <InvButton onClick={onOpen}>Open Modal</InvButton>

      <InvModal isOpen={isOpen} onClose={onClose}>
        <InvModalOverlay />
        <InvModalContent>
          <InvModalHeader>Modal Title</InvModalHeader>
          <InvModalCloseButton />
          <InvModalBody>
            Slices of banana are caramelized with brown sugar and butter, then
            rolled in sushi rice and topped with a drizzle of caramel sauce.
            This variety offers a sweet and rich flavor, combining the
            creaminess of banana with the indulgent taste of caramel.
          </InvModalBody>

          <InvModalFooter>
            <InvButton colorScheme="base" mr={3} onClick={onClose}>
              Close
            </InvButton>
            <InvButton colorScheme="green" variant="ghost">
              Secondary Action
            </InvButton>
          </InvModalFooter>
        </InvModalContent>
      </InvModal>
    </>
  );
};

export const Default: Story = {
  render: Component,
};
