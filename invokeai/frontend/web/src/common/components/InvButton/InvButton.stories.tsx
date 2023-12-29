import { Flex } from '@chakra-ui/layout';
import type { Meta, StoryObj } from '@storybook/react';
import { InvHeading } from 'common/components/InvHeading/wrapper';
import { IoSparkles } from 'react-icons/io5';

import { InvButton } from './InvButton';
import type { InvButtonProps } from './types';

const meta: Meta<typeof InvButton> = {
  title: 'Primitives/InvButton',
  tags: ['autodocs'],
  component: InvButton,
  parameters: {
    controls: { expanded: true },
  },
  argTypes: {
    isLoading: {
      defaultValue: false,
      control: { type: 'boolean' },
    },
    isDisabled: {
      defaultValue: false,
      control: { type: 'boolean' },
    },
  },
};

export default meta;
type Story = StoryObj<typeof InvButton>;

const colorSchemes = ['base', 'invokeYellow', 'red', 'green', 'blue'] as const;
const variants = ['solid', 'outline', 'ghost', 'link'] as const;
const sizes = ['xs', 'sm', 'md', 'lg'] as const;

const Component = (props: InvButtonProps) => {
  return (
    <Flex gap={4} flexDir="column">
      {sizes.map((size) => (
        <>
          <InvHeading>Size: {size}</InvHeading>

          <Flex key={size} gap={4} flexDir="column">
            {colorSchemes.map((colorScheme) => (
              <Flex key={colorScheme} gap={4}>
                {variants.map((variant) => (
                  <>
                    <InvButton
                      size={size}
                      key={`${variant}${colorScheme}`}
                      variant={variant}
                      colorScheme={colorScheme}
                      {...props}
                    >
                      {variant}
                    </InvButton>
                    {['solid', 'outline'].includes(variant) && (
                      <InvButton
                        size={size}
                        key={`${variant}${colorScheme}leftIcon`}
                        variant={variant}
                        colorScheme={colorScheme}
                        leftIcon={<IoSparkles />}
                        {...props}
                      >
                        {variant}
                      </InvButton>
                    )}
                    {['solid', 'outline'].includes(variant) && (
                      <InvButton
                        size={size}
                        key={`${variant}${colorScheme}rightIcon`}
                        variant={variant}
                        colorScheme={colorScheme}
                        rightIcon={<IoSparkles />}
                        {...props}
                      >
                        {variant}
                      </InvButton>
                    )}
                  </>
                ))}
              </Flex>
            ))}
          </Flex>
        </>
      ))}
    </Flex>
  );
};

export const Default: Story = {
  render: Component,
};
